import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import random

from dotenv import load_dotenv

logger = logging.getLogger("api_client")
logger.setLevel(logging.INFO)

try:
    from pymongo import MongoClient
    from openai import OpenAI
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.warning(f"Failed to import one or more dependencies: {e}")
    IMPORTS_SUCCESSFUL = False

class SimpleAPIClient:
    def __init__(self, system_prompt: Optional[str] = None):
        load_dotenv()
        
        if not IMPORTS_SUCCESSFUL:
            raise ImportError("Required dependencies not available. Please install them and try again.")
            
        self._init_client(system_prompt)
        self.curriculum_collection = self.db["Curriculum"]
        self.user_progress_collection = self.db["UserProgress"]

    def _init_client(self, system_prompt: Optional[str] = None):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file.")

        self.mongo_db_uri = os.getenv("MONGO_DB_URI")
        self.mongodb_database_name = os.getenv("MONGODB_DATABASE_NAME", "Chatbot")
        self.mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME", "chatHistory")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        
        if not self.mongo_db_uri:
            raise ValueError("MONGO_DB_URI not set in .env file.")

        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

        self.mongo_client = MongoClient(self.mongo_db_uri)
        self.mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB Atlas successfully!")
        
        self.db = self.mongo_client[self.mongodb_database_name]
        self.collection = self.db[self.mongodb_collection_name]

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def generate_embedding(self, text: str) -> List[float]:
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def save_message_to_db(self, username: str, message: Dict[str, Any], session_id: Optional[str] = None):
        try:
            # Always generate a new embedding for each message
            content = message.get("content", "")
            embedding = self.generate_embedding(content)
            
            message_doc = {
                "username": username,
                "session_id": session_id or "default",
                "timestamp": datetime.utcnow(),
                "role": message.get("role"),
                "content": content,
                "embedding": embedding,
                "metadata": message.get("metadata", {})
            }
            self.collection.insert_one(message_doc)
            logger.info(f"Saved message with embedding to MongoDB for user: {username}")
        except Exception as e:
            logger.error(f"Error saving message to MongoDB: {e}")

    def get_conversation_history(self, username: str, session_id: Optional[str] = None, limit: int = 50):
        try:
            query = {"username": username}
            if session_id:
                query["session_id"] = session_id
            
            messages = list(self.collection.find(query)
                          .sort("timestamp", 1)
                          .limit(limit))
            
            return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def get_current_topic(self, username: str, session_id: str) -> Dict:
        """Get the current topic for the user"""
        try:
            # First check for any existing progress for this username
            progress = self.user_progress_collection.find_one({
                "username": username
            })
            
            if not progress:
                # Initialize user progress with first topic
                first_topic = self.curriculum_collection.find_one({"order": 1})
                if first_topic:
                    self.user_progress_collection.insert_one({
                        "username": username,
                        "session_id": session_id,
                        "current_topic": first_topic["topic_id"],
                        "completed_topics": [],
                        "assessment_status": {
                            "topic_id": first_topic["topic_id"],
                            "attempts": 0,
                            "passed": False,
                            "last_attempt": None
                        },
                        "learning_phase_complete": False
                    })
                    return first_topic
            else:
                # If progress exists but session_id is different, update it
                if progress["session_id"] != session_id:
                    self.user_progress_collection.update_one(
                        {"username": username},
                        {"$set": {"session_id": session_id}}
                    )
                current_topic = self.curriculum_collection.find_one({
                    "topic_id": progress["current_topic"]
                })
                return current_topic
                
        except Exception as e:
            logger.error(f"Error getting current topic: {e}")
            return None

    def get_topic_content(self, topic_id: str) -> Dict:
        """Get the content for a specific topic"""
        try:
            return self.curriculum_collection.find_one({"topic_id": topic_id})
        except Exception as e:
            logger.error(f"Error getting topic content: {e}")
            return None

    def evaluate_assessment(self, username: str, session_id: str, answer: str) -> Dict:
        """Evaluate user's answer to the assessment question"""
        try:
            progress = self.user_progress_collection.find_one({
                "username": username
            })
            
            if not progress:
                return {"error": "No progress found for user"}
                
            current_topic = self.get_topic_content(progress["current_topic"])
            if not current_topic:
                return {"error": "Topic not found"}
                
            assessment = current_topic["content"]["assessment"]
            print(f"[DEBUG] User answer: {answer}")
            print(f"[DEBUG] Correct answer: {assessment['correct_answer']}")
            is_correct = self._check_answer(answer, assessment["correct_answer"])
            print(f"[DEBUG] is_correct: {is_correct}")
            
            # Update assessment status
            result = self.user_progress_collection.update_one(
                {
                    "username": username
                },
                {
                    "$inc": {"assessment_status.attempts": 1},
                    "$set": {
                        "assessment_status.last_attempt": datetime.utcnow(),
                        "assessment_status.passed": is_correct,
                        "session_id": session_id
                    }
                }
            )
            print(f"[DEBUG] MongoDB update matched: {result.matched_count}, modified: {result.modified_count}")
            
            if is_correct:
                # Move to next topic
                next_topic = self.curriculum_collection.find_one({
                    "order": current_topic["order"] + 1
                })
                
                if next_topic:
                    self.user_progress_collection.update_one(
                        {
                            "username": username
                        },
                        {
                            "$set": {
                                "current_topic": next_topic["topic_id"],
                                "assessment_status": {
                                    "topic_id": next_topic["topic_id"],
                                    "attempts": 0,
                                    "passed": False,
                                    "last_attempt": None
                                },
                                "session_id": session_id
                            },
                            "$push": {"completed_topics": current_topic["topic_id"]}
                        }
                    )
                else:
                    # No more topics, mark learning phase as complete
                    self.user_progress_collection.update_one(
                        {
                            "username": username
                        },
                        {
                            "$set": {
                                "learning_phase_complete": True,
                                "session_id": session_id
                            }
                        }
                    )
            
            return {
                "is_correct": is_correct,
                "hints": assessment["hints"] if not is_correct else [],
                "next_topic": next_topic["topic_id"] if is_correct and next_topic else None,
                "learning_complete": is_correct and not next_topic
            }
            
        except Exception as e:
            print(f"[DEBUG] Error in evaluate_assessment: {e}")
            logger.error(f"Error evaluating assessment: {e}")
            return {"error": str(e)}

    def _check_answer(self, user_answer: str, correct_answer: str) -> bool:
        """Evaluate user's answer using LLM for semantic understanding"""
        try:
            evaluation_prompt = f"""You are an educational assessment evaluator. Your task is to evaluate if the student's answer demonstrates understanding of the concept, even if the wording is different.\n\nCorrect Answer: {correct_answer}\nStudent's Answer: {user_answer}\n\nEvaluate if the student's answer:\n1. Contains the key concepts from the correct answer\n2. Demonstrates understanding of the topic\n3. Is logically sound and relevant\n\nRespond with ONLY 'CORRECT' if the answer is acceptable, or 'INCORRECT' if it's not.\nBe lenient in your evaluation - if the student demonstrates understanding even with different wording, mark it as CORRECT."""
            response = self.openai_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "system", "content": evaluation_prompt}],
                temperature=0.3,
                max_tokens=10
            )
            evaluation = response.choices[0].message.content.strip().upper()
            logger.info(f"[DEBUG] LLM output: {evaluation}")
            return evaluation == "CORRECT"
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return self._fallback_keyword_check(user_answer, correct_answer)

    def _fallback_keyword_check(self, user_answer: str, correct_answer: str) -> bool:
        """Fallback method using keyword matching when LLM evaluation fails"""
        try:
            # Extract key concepts from correct answer
            extraction_prompt = f"""Extract the key concepts from this answer. Return only the key concepts as a comma-separated list:

            Answer: {correct_answer}"""

            response = self.openai_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "system", "content": extraction_prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            key_concepts = [concept.strip().lower() for concept in response.choices[0].message.content.split(",")]
            user_answer_lower = user_answer.lower()
            
            # Check if at least 50% of key concepts are present in user's answer
            matches = sum(1 for concept in key_concepts if concept in user_answer_lower)
            return matches / len(key_concepts) >= 0.5
            
        except Exception as e:
            logger.error(f"Error in fallback keyword check: {e}")
            return False

    def is_learning_phase_complete(self, username: str, session_id: str) -> bool:
        """Check if user has completed the learning phase"""
        try:
            progress = self.user_progress_collection.find_one({
                "username": username
            })
            return progress.get("learning_phase_complete", False) if progress else False
        except Exception as e:
            logger.error(f"Error checking learning phase completion: {e}")
            return False

    def _detect_intent(self, user_message: str) -> str:
        msg = user_message.lower().strip()
        if any(kw in msg for kw in ["all topics", "curriculum", "list topics", "show topics"]):
            return "list_topics"
        if any(kw in msg for kw in ["next", "what do you suggest", "what's next", "continue", "move on"]):
            return "next_topic"
        # Check for topic name in message
        topics = list(self.curriculum_collection.find({}, {"topic_name": 1, "topic_id": 1}))
        for t in topics:
            if t["topic_name"].lower() in msg:
                return f"jump_to:{t['topic_id']}"
        return "default"

    def _list_all_topics(self):
        topics = list(self.curriculum_collection.find({}, {"order": 1, "topic_name": 1}).sort("order", 1))
        return "Here are all the topics in the curriculum:\n" + "\n".join([f"{t['order']}. {t['topic_name']}" for t in topics])

    def set_awaiting_assessment(self, username: str, session_id: str, value: bool):
        try:
            self.user_progress_collection.update_one(
                {"username": username},
                {
                    "$set": {
                        "awaiting_assessment": value,
                        "session_id": session_id
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error setting awaiting_assessment: {e}")

    def is_awaiting_assessment(self, username: str, session_id: str) -> bool:
        try:
            progress = self.user_progress_collection.find_one({"username": username})
            return progress.get("awaiting_assessment", False) if progress else False
        except Exception as e:
            logger.error(f"Error checking awaiting_assessment: {e}")
            return False

    def get_next_scenario(self, username: str, session_id: str) -> dict:
        try:
            progress = self.user_progress_collection.find_one({"username": username})
            discussed = progress.get("discussed_scenarios", []) if progress else []
            all_ids = [str(i) for i in range(1, 21)]
            available_ids = [sid for sid in all_ids if sid not in discussed]
            if not available_ids:
                return None
            # Ensure random selection is always an integer string
            scenario_id = random.choice(available_ids)
            if isinstance(scenario_id, float):
                scenario_id = str(int(round(scenario_id)))
            scenario = self.db["Scenarios"].find_one({"scenario_id": scenario_id})
            if scenario:
                self.user_progress_collection.update_one(
                    {"username": username},
                    {
                        "$push": {"discussed_scenarios": scenario["scenario_id"]},
                        "$set": {
                            "current_scenario": scenario["scenario_id"],
                            "session_id": session_id
                        }
                    }
                )
            return scenario
        except Exception as e:
            logger.error(f"Error getting next scenario: {e}")
            return None

    def get_current_scenario(self, username: str, session_id: str) -> dict:
        try:
            progress = self.user_progress_collection.find_one({"username": username})
            scenario_id = progress.get("current_scenario") if progress else None
            if scenario_id:
                return self.db["Scenarios"].find_one({"scenario_id": scenario_id})
            return self.get_next_scenario(username, session_id)
        except Exception as e:
            logger.error(f"Error getting current scenario: {e}")
            return None

    def find_similar_messages_by_embedding(self, query_embedding: list, username: str, limit: int = 5) -> list:
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": limit
                    }
                },
                {
                    "$match": {
                        "username": username
                    }
                },
                {
                    "$project": {
                        "role": 1,
                        "content": 1,
                        "timestamp": 1,
                        "score": { "$meta": "vectorSearchScore" }
                    }
                }
            ]
            similar_messages = list(self.collection.aggregate(pipeline))
            return similar_messages
        except Exception as e:
            logger.error(f"Error finding similar messages: {e}")
            return []

    def get_relevant_messages(self, query: str, username: str, limit: int = 3) -> list:
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return []
        similar_messages = self.find_similar_messages_by_embedding(query_embedding, username, limit)
        return [msg["content"] for msg in similar_messages]

    def generate_response(self, query: str, username: str, session_id: str, conversation_history: Optional[List[Dict]] = None) -> str:
        logger.info("Entered generate_response function.")
        if not query:
            logger.warning("Empty query received.")
            return "Query cannot be empty."
        try:
            relevant_msgs = self.get_relevant_messages(query, username, limit=10)
            print(f"[RAG] Query: {query}")
            print(f"[RAG] Relevant messages retrieved: {relevant_msgs}")
            instruction = "You have access to relevant messages from the user's past. Use this information to answer questions, including those about the user's name, if the information is present."
            # Learning phase
            if not self.is_learning_phase_complete(username, session_id):
                current_topic = self.get_current_topic(username, session_id)
                if current_topic:
                    if self.is_awaiting_assessment(username, session_id):
                        evaluation = self.evaluate_assessment(username, session_id, query)
                        self.set_awaiting_assessment(username, session_id, False)
                        if evaluation.get("error"):
                            assistant_response = f"Error: {evaluation['error']}"
                        elif evaluation["is_correct"]:
                            next_topic = self.curriculum_collection.find_one({"order": current_topic["order"] + 1})
                            if next_topic:
                                assistant_response = f"Correct! Let's move on to the next topic: {next_topic['topic_name']}\n\n" + "\n".join(next_topic['content']['main_points'])
                            else:
                                assistant_response = "Congratulations! You have completed the learning phase. You can now proceed to the discussion phase."
                        else:
                            hints = "\n".join(evaluation["hints"])
                            assistant_response = f"That's not quite right. Here are some hints:\n{hints}\n\nPlease try again."
                        self.save_message_to_db(username, {"role": "user", "content": query}, session_id)
                        self.save_message_to_db(username, {"role": "assistant", "content": assistant_response}, session_id)
                        return assistant_response
                    system_prompt = f"""{instruction}
You are a teaching assistant helping the user learn about fine-tuning. 
Current topic: {current_topic['topic_name']}
Main points to cover: {', '.join(current_topic['content']['main_points'])}
Examples to use: {', '.join(current_topic['content']['examples'])}

Your role is to:
1. Explain the current topic clearly and systematically
2. Use the provided examples to illustrate concepts
3. Answer any questions the user has about the topic
4. When the user seems ready, ask them the assessment question: {current_topic['content']['assessment']['question']}

Be friendly and encouraging in your explanations."""
                    if relevant_msgs:
                        system_prompt += "\n\nRelevant messages from the user in the past:\n" + "\n".join(f"- {msg}" for msg in relevant_msgs)
                    messages = [{"role": "system", "content": system_prompt}]
                    if conversation_history:
                        messages.extend(conversation_history[-5:])
                    messages.append({"role": "user", "content": query})
                    response = self.openai_client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    assistant_response = response.choices[0].message.content
                    assessment_question = current_topic['content']['assessment']['question'].strip().lower()
                    if assessment_question in assistant_response.strip().lower():
                        self.set_awaiting_assessment(username, session_id, True)
                    self.save_message_to_db(username, {"role": "user", "content": query}, session_id)
                    self.save_message_to_db(username, {"role": "assistant", "content": assistant_response}, session_id)
                    return assistant_response
            # Discussion phase
            if self.is_learning_phase_complete(username, session_id):
                scenario = self.get_current_scenario(username, session_id)
                if not scenario:
                    assistant_response = "There are no more scenarios to discuss. Feel free to ask any other questions!"
                else:
                    system_prompt = f"""{instruction}
You are now a Socratic discussion partner. Engage the user in open-ended, thoughtful discussion about fine-tuning in real-world scenarios. Ask probing questions, encourage critical thinking, and help the user reason through different approaches. Use real-life examples and adapt your questions to the user's context.

Current Scenario: {scenario['title']}
Description: {scenario['description']}
Tags: {', '.join(scenario['tags'])}

Begin the discussion by asking the user an open-ended question about this scenario, and continue the conversation in a Socratic, exploratory manner."""
                    if relevant_msgs:
                        system_prompt += "\n\nRelevant messages from the user in the past:\n" + "\n".join(f"- {msg}" for msg in relevant_msgs)
                    messages = [{"role": "system", "content": system_prompt}]
                    if conversation_history:
                        messages.extend(conversation_history[-10:])
                    messages.append({"role": "user", "content": query})
                    response = self.openai_client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    assistant_response = response.choices[0].message.content
                self.save_message_to_db(username, {"role": "user", "content": query}, session_id)
                self.save_message_to_db(username, {"role": "assistant", "content": assistant_response}, session_id)
                return assistant_response
            # Fallback (should not be reached)
            return "Something went wrong. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Apologies, an error occurred: {str(e)}"

    def generate_chat_summary(self, session_id: str, username: str, num_messages: int = 10) -> Optional[str]:
        logger.info(f"Entered generate_chat_summary with session_id: {session_id}, username: {username}")
        if not session_id or session_id == "default" or not username:
            logger.warning("Invalid session_id or username for summary generation.")
            return None

        try:
            logger.info("Fetching recent messages for summary.")
            messages = list(self.collection.find(
                {"session_id": session_id, "username": username}
            ).sort("timestamp", -1).limit(num_messages))
            
            if not messages:
                logger.info("No messages found for summary generation.")
                return None

            messages.reverse()
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            summary_prompt = """Please provide a concise summary of the following conversation. 
            Focus on the main topics discussed and any important decisions or conclusions reached. 
            Keep the summary under 200 words.
            
            Conversation:
            {text}"""
            
            logger.info("Calling OpenAI for summary generation.")
            summary = self.openai_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "system", "content": summary_prompt.format(text=text)}],
                temperature=0.3,
                max_tokens=200
            ).choices[0].message.content
            
            logger.info(f"Summary generated: {summary[:100]}")  # Log first 100 chars

            # Store summary in a separate collection
            summary_collection = self.db["Summaries"]
            summary_collection.update_one(
                {"session_id": session_id, "username": username},
                {
                    "$set": {
                        "summary": summary,
                        "updated_at": datetime.utcnow(),
                        "message_count": len(messages)
                    }
                },
                upsert=True
            )
            logger.info("Summary stored successfully.")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating or storing summary: {e}", exc_info=True)
            return None

    def get_session_summary(self, session_id: str, username: str) -> str:
        logger.info(f"Entered get_session_summary with session_id: {session_id}, username: {username}")
        if not session_id or session_id == "default" or not username:
            logger.warning("Invalid session_id or username for summary retrieval.")
            return ""
            
        try:
            summary_collection = self.db["Summaries"]
            doc = summary_collection.find_one({"session_id": session_id, "username": username})
            
            if doc and "summary" in doc:
                logger.info("Summary found in database.")
                return doc["summary"]
            else:
                logger.info("No summary found in database, generating new summary.")
                return self.generate_chat_summary(session_id, username) or ""
                
        except Exception as e:
            logger.error(f"Error retrieving summary from database: {e}", exc_info=True)
            return ""

    def should_update_summary(self, session_id: str, username: str) -> bool:
        try:
            summary_collection = self.db["Summaries"]
            doc = summary_collection.find_one({"session_id": session_id, "username": username})
            
            if not doc:
                return True
                
            # Get current message count
            current_count = self.collection.count_documents({
                "session_id": session_id,
                "username": username
            })
            
            # Update if message count has increased by more than 3
            return current_count - doc.get("message_count", 0) > 3
            
        except Exception as e:
            logger.error(f"Error checking if summary should be updated: {e}")
            return False

if __name__ == '__main__':
    print("Initializing SimpleAPIClient...")
    print("Ensure .env file has: OPENAI_API_KEY, MONGO_DB_URI")
    
    try:
        client = SimpleAPIClient()
        print("SimpleAPIClient initialized.")

        print("\nTesting response generation...")
        test_query = "Hello, how are you?"
        response = client.generate_response(test_query, "test_user", "default")
        print(f"User Query: {test_query}")
        print(f"LLM Response: {response}")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
        print("Please check your .env configuration and MongoDB connection.")
