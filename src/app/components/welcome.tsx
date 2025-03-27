import React from "react";
import { FiSend } from "react-icons/fi"; // using an icon from react-icons
import styles from "../styles/homepage.module.css";

interface WelcomeProps {
  input: string;
  onInputChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
}

function Welcome({ input, onInputChange, onSubmit }: WelcomeProps) {
  return (
    <div className={styles.welcomeContainer}>
      <h1 className={styles.welcomeTitle}>Welcome to Socratic AI</h1>
      <p className={styles.welcomeSubtitle}>
        What would you like to learn today?
      </p>
      <form
        onSubmit={onSubmit}
        className={`${styles.form} ${styles.welcomeInput}`}
      >
        <input
          type="text"
          value={input}
          onChange={onInputChange}
          placeholder="Start typing your question here..."
          className={styles.input} // styling is applied via CSS module
        />
        <button type="submit" className={styles.button}>
          <FiSend className={styles.icon} /> {/* Icon styled via CSS */}
          Send
        </button>
      </form>
    </div>
  );
}

export default Welcome;
