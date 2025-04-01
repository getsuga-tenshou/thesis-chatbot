import mongoose from "mongoose";

const ConversationSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
  },
  messages: [
    {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Message",
    },
  ],
  createdAt: {
    type: Date,
    default: Date.now,
  },
  lastUpdated: {
    type: Date,
    default: Date.now,
  },
});

const Conversation =
  mongoose.models.Conversation ||
  mongoose.model("Conversation", ConversationSchema);

export { Conversation };
