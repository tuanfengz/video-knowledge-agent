import { useState, useRef } from "react";
import styles from "./SearchPanel.module.css";

const SUGGESTED = [
  "Indoor scene with laptops and coffee",
  "Vehicles on a highway at night",
  "A Nike backpack",
  "Person using a cell phone",
  "Sports activity outdoors",
];

export default function SearchPanel() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expandedIdx, setExpandedIdx] = useState(null);
  const videoRefs = useRef({});

  async function handleSearch(q) {
    const text = (q ?? query).trim();
    if (!text) return;
    setLoading(true);
    setError("");
    setResults(null);
    setExpandedIdx(null);
    if (q) setQuery(q);

    try {
      const params = new URLSearchParams({ q: text, top_k: topK });
      const res = await fetch(`/search?${params}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.card}>
      <h2>Search Video Segments</h2>
      <p className={styles.subtitle}>
        Find specific moments across all indexed videos using a natural-language query.
      </p>

      {/* Suggested queries */}
      {!results && (
        <div className={styles.suggestions}>
          {SUGGESTED.map((s) => (
            <button key={s} className={styles.chip} onClick={() => handleSearch(s)}>
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input row */}
      <div className={styles.inputRow}>
        <input
          className={styles.input}
          type="text"
          placeholder="e.g. indoor scene with laptop and coffee…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          disabled={loading}
        />
        <select
          className={styles.select}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          disabled={loading}
        >
          {[5, 10, 20, 50].map((n) => (
            <option key={n} value={n}>Top {n}</option>
          ))}
        </select>
        <button
          className={styles.searchBtn}
          onClick={() => handleSearch()}
          disabled={loading || !query.trim()}
        >
          {loading ? "…" : "Search"}
        </button>
      </div>

      {error && <p className={styles.error}>{error}</p>}

      {/* Results */}
      {results && (
        <div className={styles.results}>
          <p className={styles.meta}>
            {results.results.length} results from{" "}
            <strong>{results.total_segments_searched.toLocaleString()}</strong> indexed segments
            {" "}for <em>"{results.query}"</em>
          </p>

          {results.results.length === 0 ? (
            <p className={styles.empty}>No matching segments found.</p>
          ) : (
            <ul className={styles.list}>
              {results.results.map((r, idx) => {
                const isOpen = expandedIdx === idx;
                const videoSrc = `/video/raw/${encodeURIComponent(r.video_id)}`;

                function handleToggle() {
                  if (isOpen) {
                    setExpandedIdx(null);
                  } else {
                    setExpandedIdx(idx);
                    // Seek to segment start after the video element mounts/loads
                    requestAnimationFrame(() => {
                      const vid = videoRefs.current[idx];
                      if (vid) {
                        vid.currentTime = r.start_s;
                        vid.play().catch(() => {});
                      }
                    });
                  }
                }

                function handleVideoLoaded(e) {
                  e.target.currentTime = r.start_s;
                  e.target.play().catch(() => {});
                }

                function handleTimeUpdate(e) {
                  // Pause at segment end (with 0.1s buffer to catch tick overshoots).
                  if (e.target.currentTime >= r.end_s - 0.1) {
                    e.target.pause();
                    e.target.currentTime = r.start_s;
                  }
                }

                function handleEnded(e) {
                  // Fallback: video file ended before timeupdate caught it.
                  e.target.currentTime = r.start_s;
                }

                return (
                  <li key={`${r.video_id}-${r.start_s}`} className={styles.item}>
                    <div className={styles.itemHeader} onClick={handleToggle} style={{ cursor: "pointer" }}>
                      <span className={styles.rank}>#{r.rank}</span>
                      <span className={styles.score}>score {r.score.toFixed(3)}</span>
                      <span className={styles.videoId}>{r.video_id}</span>
                      <span className={styles.timestamp}>
                        {r.start_s.toFixed(1)}s – {r.end_s.toFixed(1)}s
                      </span>
                      <span className={styles.playHint}>{isOpen ? "▲ hide" : "▶ play"}</span>
                    </div>

                    <p className={styles.description}>{r.description}</p>

                    {r.labels && Object.keys(r.labels).length > 0 && (
                      <div className={styles.labels}>
                        {Object.entries(r.labels)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 6)
                          .map(([lbl, cnt]) => (
                            <span key={lbl} className={styles.label}>
                              {lbl} ({cnt})
                            </span>
                          ))}
                      </div>
                    )}

                    {isOpen && (
                      <div className={styles.videoWrapper}>
                        <video
                          ref={(el) => { videoRefs.current[idx] = el; }}
                          className={styles.video}
                          src={videoSrc}
                          controls
                          onLoadedMetadata={handleVideoLoaded}
                          onTimeUpdate={handleTimeUpdate}
                          onEnded={handleEnded}
                        />
                        <p className={styles.videoHint}>
                          Segment: {r.start_s.toFixed(1)}s – {r.end_s.toFixed(1)}s (loops within window)
                        </p>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}

          <button className={styles.clearBtn} onClick={() => setResults(null)}>
            ← New search
          </button>
        </div>
      )}
    </div>
  );
}
