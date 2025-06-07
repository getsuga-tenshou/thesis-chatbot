# EthosBot
## About

This project was developed as part of a thesis research to investigate whether pedagogical conversational agents can effectively improve ethical reasoning among students. By combining AI technology with structured learning approaches, EthosBot aims to enhance students' ability to think critically about ethical dilemmas and develop stronger moral reasoning skills.

## Overview

EthosBot uses advanced AI to engage users in meaningful conversations about ethics, breaking down complex topics into understandable concepts while encouraging critical thinking and ethical reasoning. The application features a clean, user-friendly interface where users can have natural conversations with the AI assistant.

## Key Features

- Interactive chat interface
- User authentication system
- Persistent chat history
- Ethical learning framework
- Admin dashboard for user management

## Tech Stack

- Frontend: Streamlit
- Backend: Python
- Database: MongoDB
- AI: OpenAI GPT

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run: `streamlit run main.py`

For detailed setup instructions and configuration, please refer to the documentation in the repository.

## Features

- **User Authentication**: Secure login/signup system with encrypted password storage
- **Chat Interface**: Clean, intuitive chat interface powered by OpenAI's GPT models
- **Persistent Storage**: Chat history stored in MongoDB for continuity across sessions
- **Session Management**: Individual chat sessions with unique identifiers
- **Admin Panel**: Administrative interface for user management

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: MongoDB (for user data and chat history), SQLite (fallback for auth)
- **AI Model**: OpenAI GPT (configurable model)
- **Authentication**: Custom auth system with encryption

## Setup

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (or local MongoDB instance)
- OpenAI API key

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd EthosBot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
MONGO_DB_URI=your_mongodb_connection_string
MONGODB_DATABASE_NAME=Chatbot
MONGODB_COLLECTION_NAME=chatHistory
MONGODB_USERS_COLLECTION=userData
LLM_MODEL_NAME=gpt-4o-mini
```

4. Run the application:

```bash
streamlit run main.py
```

## Usage

1. **First Time Setup**: Create an account using the signup page
2. **Login**: Use your credentials to access the chat interface
3. **Chat**: Start conversing with the AI assistant
4. **Session Management**: Use "New Chat" to start fresh conversations
5. **History**: Load previous chat history to continue conversations

## File Structure

```
EthosBot/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── utils/
│   ├── auth_setup.py      # Authentication system
│   ├── api_client.py      # OpenAI and MongoDB client  
├── pages/
│   ├── Login.py      
│   ├── Signup.py     
│   └── Admin.py      
└── db/
    └── auth.db            # SQLite database (fallback)
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `MONGO_DB_URI`: MongoDB connection string
- `MONGODB_DATABASE_NAME`: Database name (default: "Chatbot")
- `MONGODB_COLLECTION_NAME`: Collection for chat messages (default: "chatHistory")
- `MONGODB_USERS_COLLECTION`: Collection for user data (default: "userData")
- `LLM_MODEL_NAME`: OpenAI model to use (default: "gpt-4o-mini")

### Admin Setup

To create an admin user, you can either:

1. Use the admin panel if you have access
2. Manually set the `su` field to `1` in the user document in MongoDB

## Security Features

- Password encryption using bcrypt
- Secure session management
- Environment variable protection for sensitive data
- MongoDB connection security


## License

This project is licensed under the MIT License.

