import SearchPanel from "./components/SearchPanel.jsx";
import styles from "./App.module.css";

export default function App() {
  return (
    <div className={styles.layout}>
      <header className={styles.header}>
        <h1>Video Knowledge Agent</h1>
        <p>Search across 130+ indexed videos by natural language.</p>
      </header>

      <main className={styles.main}>
        <div className={styles.fullRow}>
          <SearchPanel />
        </div>
      </main>
    </div>
  );
}
