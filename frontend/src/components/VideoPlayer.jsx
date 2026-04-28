import styles from "./VideoPlayer.module.css";

export default function VideoPlayer({ src, title, badge }) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <span className={styles.title}>{title}</span>
        {badge && <span className={styles.badge}>{badge}</span>}
      </div>
      <video
        className={styles.video}
        src={src}
        controls
        playsInline
        preload="metadata"
      />
    </div>
  );
}
