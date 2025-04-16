import React from "react";
import { FiSend } from "react-icons/fi";
import styles from "../styles/homepage.module.css";

interface WelcomeProps {
  input: string;
  onInputChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
}

function Welcome({ input, onInputChange, onSubmit }: WelcomeProps) {
  return (
    <div
      className={styles.welcomeContainer}
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        padding: "2rem",
        textAlign: "center",
        flex: 1,
        background: "linear-gradient(to bottom right, #111, #222)",
      }}
    >
      <h1 className={styles.welcomeTitle}>Welcome to Socratic AI</h1>
      <p className={styles.welcomeSubtitle}>
        What would you like to learn today?
      </p>
      <form
        onSubmit={onSubmit}
        style={{
          display: "flex",
          gap: "0.5rem",
          width: "100%",
          maxWidth: "500px",
        }}
      >
        <input
          type="text"
          value={input}
          onChange={onInputChange}
          placeholder="Start typing your question here..."
          style={{
            flexGrow: 1,
            padding: "0.5rem",
            borderRadius: "0.25rem",
            border: "1px solid #444",
            outline: "none",
            backgroundColor: "#222",
            color: "#fff",
          }}
        />
        <button
          type="submit"
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.25rem",
            border: "none",
            backgroundColor: "#8a2be2",
            color: "#fff",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            fontSize: "0.9rem",
          }}
        >
          <FiSend />
          <span>Send</span>
        </button>
      </form>
    </div>
  );
}

export default Welcome;
