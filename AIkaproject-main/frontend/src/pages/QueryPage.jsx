import { useEffect, useMemo, useState } from 'react';
import {
  Loader2,
  LayoutList,
  Search,
  BarChart3,
  PanelLeftClose,
  PanelLeftOpen,
  MousePointerClick,
  ChevronDown,
  ChevronRight,
  Sparkles,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchPyqList, queryByPyqId } from '../services/api';
import '../styles/QueryPage.css';

/* ——— colour palette for charts ——— */
const CHART_COLORS = ['#818cf8', '#a78bfa', '#67e8f9', '#34d399', '#fbbf24', '#fb7185'];

/* ——— custom bar chart tooltip ——— */
const ChartTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      style={{
        background: '#161b22',
        border: '1px solid #30363d',
        borderRadius: 10,
        padding: '10px 14px',
        boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
        maxWidth: 260,
      }}
    >
      <p style={{ color: '#e6edf3', fontWeight: 600, fontSize: 13, marginBottom: 4 }}>
        {d.label} — {d.file_name}
      </p>
      <p style={{ color: '#67e8f9', fontSize: 12, marginBottom: 6 }}>
        Score: {d.score.toFixed(4)}
      </p>
      <p style={{ color: '#8b949e', fontSize: 11, lineHeight: 1.5 }}>
        {d.text?.slice(0, 120)}…
      </p>
    </div>
  );
};

/* ——— custom pie chart label ——— */
const renderPieLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, name, percent }) => {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  if (percent < 0.06) return null;
  return (
    <text x={x} y={y} fill="#e6edf3" fontSize={11} fontWeight={600} textAnchor="middle" dominantBaseline="central">
      {name}
    </text>
  );
};

export default function QueryPage() {
  /* —— state —— */
  const [pyqs, setPyqs] = useState([]);
  const [selectedPyq, setSelectedPyq] = useState(null);
  const [result, setResult] = useState(null);
  const [loadingPyqs, setLoadingPyqs] = useState(true);
  const [loadingResult, setLoadingResult] = useState(false);
  const [error, setError] = useState('');
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedCandidates, setExpandedCandidates] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [focusedIdx, setFocusedIdx] = useState(-1);

  /* —— fetch PYQs on mount —— */
  useEffect(() => {
    const loadPyqs = async () => {
      setLoadingPyqs(true);
      setError('');
      try {
        const data = await fetchPyqList(500);
        setPyqs(data.items || []);
      } catch (err) {
        setError(err.message || 'Failed to load PYQ list.');
      } finally {
        setLoadingPyqs(false);
      }
    };
    loadPyqs();
  }, []);



  /* —— select PYQ —— */
  const handleSelectPyq = async (item) => {
    setSelectedPyq(item);
    setLoadingResult(true);
    setError('');
    setExpandedCandidates({});
    try {
      const data = await queryByPyqId(item.pyq_id);
      setResult(data);
      setSelectedCandidate(data.chart?.length > 0 ? data.chart[0] : null);
    } catch (err) {
      setError(err.message || 'Failed to retrieve top paragraphs.');
      setResult(null);
      setSelectedCandidate(null);
    } finally {
      setLoadingResult(false);
    }
  };

  /* —— candidate click —— */
  const handleCandidateSelect = (candidate) => {
    if (!candidate) return;
    setSelectedCandidate(candidate);
  };

  /* —— toggle C1/C2/C3 expansion —— */
  const toggleCandidate = (chunkId) => {
    setExpandedCandidates((prev) => ({
      ...prev,
      [chunkId]: !prev[chunkId],
    }));
  };

  /* —— pie data (normalised score shares) —— */
  const pieData = useMemo(() => {
    if (!result?.chart?.length) return [];
    const total = result.chart.reduce((sum, c) => sum + Math.max(c.score, 0.001), 0);
    return result.chart.map((c, i) => ({
      name: c.label,
      value: Math.max(c.score, 0.001) / total,
      score: c.score,
      fill: CHART_COLORS[i % CHART_COLORS.length],
    }));
  }, [result]);

  const filteredPyqs = useMemo(
    () =>
      pyqs.filter((q) => {
        if (!searchQuery) return true;
        const needle = searchQuery.toLowerCase();
        return (
          q.text?.toLowerCase().includes(needle) ||
          q.file_name?.toLowerCase().includes(needle)
        );
      }),
    [pyqs, searchQuery]
  );

  const groupedPyqs = useMemo(() => {
    const groups = {};
    filteredPyqs.forEach((q) => {
      const key = q.file_name || 'Unknown';
      if (!groups[key]) groups[key] = [];
      groups[key].push(q);
    });
    return groups;
  }, [filteredPyqs]);

  const handleSidebarKeyDown = (e) => {
    if (!filteredPyqs.length) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setFocusedIdx((i) => Math.min(i + 1, filteredPyqs.length - 1));
      return;
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setFocusedIdx((i) => Math.max(i - 1, 0));
      return;
    }

    if (e.key === 'Enter' && focusedIdx >= 0) {
      e.preventDefault();
      handleSelectPyq(filteredPyqs[focusedIdx]);
    }
  };

  /* ================================================================== */
  return (
    <div className="query-layout">
      {/* ——— Sidebar ——— */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <div className="sidebar-title-row">
            <span className="sidebar-title">
              <LayoutList size={16} />
              PYQ Questions
            </span>
            <button
              className="sidebar-close-btn"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close sidebar"
            >
              <PanelLeftClose size={16} />
            </button>
          </div>
        </div>

        <p className="sidebar-count">{pyqs.length} questions</p>

        <div className="sidebar-search">
          <Search size={14} />
          <input
            type="text"
            placeholder="Search questions..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setFocusedIdx(-1);
            }}
          />
        </div>

        <div
          className="sidebar-list"
          role="listbox"
          tabIndex={0}
          onKeyDown={handleSidebarKeyDown}
        >
          {loadingPyqs ? (
            <div className="loading-bar" style={{ justifyContent: 'center', padding: '32px 0' }}>
              <Loader2 size={16} />
              Loading…
            </div>
          ) : pyqs.length === 0 ? (
            <div className="sidebar-empty">
              <p>No PYQs indexed yet.</p>
              <p>Run the ingestion pipeline to add questions.</p>
            </div>
          ) : (
            Object.entries(groupedPyqs).map(([fileName, items]) => (
              <div key={fileName} className="pyq-group">
                <p className="pyq-group-label">{fileName}</p>
                {items.map((item) => {
                  const isActive = selectedPyq?.pyq_id === item.pyq_id;
                  const isFocused =
                    focusedIdx >= 0 &&
                    filteredPyqs[focusedIdx]?.pyq_id === item.pyq_id;
                  return (
                    <button
                      key={item.pyq_id}
                      className={`pyq-item ${isActive ? 'active' : ''} ${isFocused ? 'focused' : ''}`}
                      onClick={() => handleSelectPyq(item)}
                    >
                      <p className="pyq-item-source">
                        <span className="dot" />
                        {item.file_name || 'PYQ Source'}
                      </p>
                      <p className="pyq-item-text">{item.text}</p>
                    </button>
                  );
                })}
              </div>
            ))
          )}
        </div>
      </aside>

      {/* ——— Sidebar toggle button when closed ——— */}
      {!sidebarOpen && (
        <button
          className="sidebar-toggle"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open sidebar"
        >
          <PanelLeftOpen size={18} />
        </button>
      )}

      {/* ——— Main content ——— */}
      <div className="main-content">
        {/* Top bar */}
        <header className="top-bar">
          <div>
            <div className="top-bar-title">
              <span className="logo-icon">⚡</span>
              <h1>PYQ → NCERT Explainability Console</h1>
            </div>
            <p className="top-bar-subtitle">
              Select a PYQ from the sidebar, view top-ranked NCERT paragraphs, and explore score evidence.
            </p>
          </div>
          <div className="top-bar-badge">
            <span className="pulse-dot" />
            System Active
          </div>
        </header>

        {/* Error bar */}
        {error && (
          <div style={{ padding: '0 28px', paddingTop: 12 }}>
            <div
              style={{
                padding: '10px 16px',
                borderRadius: 10,
                background: 'rgba(251,113,133,0.08)',
                border: '1px solid rgba(251,113,133,0.2)',
                color: '#fb7185',
                fontSize: '0.8125rem',
                animation: 'fadeInUp 0.3s ease',
              }}
            >
              {error}
            </div>
          </div>
        )}

        {/* Content grid */}
        <div className="content-grid">
          {/* ———— Center panel ———— */}
          <div className="center-panel">
            {!selectedPyq && (
              <div className="center-empty">
                <MousePointerClick size={48} />
                <h3>No PYQ Selected</h3>
                <p>Click a question from the sidebar to retrieve the most relevant NCERT paragraphs.</p>
              </div>
            )}

            {selectedPyq && (
              <div className="animate-fadeIn">
                {/* Selected PYQ */}
                <div className="selected-pyq-card">
                  <p className="selected-pyq-label">Selected PYQ</p>
                  <p className="selected-pyq-text">{selectedPyq.text}</p>
                </div>

                {/* Loading */}
                {loadingResult && (
                  <div className="skeleton-list">
                    {[1, 2].map((n) => (
                      <div key={n} className="skeleton-card">
                        <div className="skeleton-line short" />
                        <div className="skeleton-line" />
                        <div className="skeleton-line" />
                        <div className="skeleton-line medium" />
                      </div>
                    ))}
                  </div>
                )}

                {/* No results */}
                {!loadingResult && result?.matches?.length === 0 && (
                  <div className="no-results">No matching NCERT paragraphs found for this PYQ.</div>
                )}

                {/* Match cards */}
                {!loadingResult &&
                  result?.matches?.map((match, idx) => (
                    <article
                      key={match.chunk_id}
                      className="match-card"
                      style={{ animationDelay: `${idx * 100}ms` }}
                    >
                      <div className="match-badges">
                        <span className="match-badge rank">Rank {match.rank}</span>
                        <span className="match-badge score">Score {match.score.toFixed(4)}</span>
                        <span className="match-badge file">{match.file_name}</span>
                      </div>

                      <div className="score-breakdown">
                        <span title="Cross-encoder">CE: {match.rerank_score?.toFixed(3)}</span>
                        <span title="Vector similarity">Vec: {match.vector_score?.toFixed(3)}</span>
                        <span title="BM25 keyword">BM25: {match.bm25_score?.toFixed(3)}</span>
                      </div>

                      <div className="score-bar-track">
                        <div
                          className="score-bar-fill"
                          style={{ width: `${Math.min(100, Math.max(5, match.score * 100))}%` }}
                        />
                      </div>

                      <p className="match-reason">{match.reason}</p>
                      <p className="match-text">{match.text}</p>
                    </article>
                  ))}
              </div>
            )}
          </div>

          {/* ———— Right panel – Charts & Insights ———— */}
          <div className="right-panel">
            <div className="right-panel-title">
              <Sparkles size={16} />
              Insights &amp; Evidence
            </div>
            <p className="right-panel-subtitle">
              Relevance scores for top candidates. Click a bar or pill to inspect.
            </p>

            {result?.chart?.length ? (
              <div className="animate-fadeIn">
                {/* Bar chart */}
                <p className="chart-section-label">Cross-Encoder Scores</p>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.chart} margin={{ left: -20, right: 8, top: 8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                      <XAxis dataKey="label" stroke="#6e7681" tick={{ fontSize: 11 }} />
                      <YAxis stroke="#6e7681" tick={{ fontSize: 11 }} domain={[0, 'dataMax + 0.05']} />
                      <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(129,140,248,0.06)' }} />
                      <Bar
                        dataKey="score"
                        radius={[6, 6, 0, 0]}
                        onClick={(data) => handleCandidateSelect(data?.payload || data)}
                        cursor="pointer"
                        animationDuration={800}
                        animationEasing="ease-out"
                      >
                        {result.chart.map((_, idx) => (
                          <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Pie chart */}
                <p className="chart-section-label">Score Distribution</p>
                <div className="pie-chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={42}
                        outerRadius={78}
                        paddingAngle={3}
                        dataKey="value"
                        labelLine={false}
                        label={renderPieLabel}
                        animationDuration={900}
                        animationEasing="ease-out"
                      >
                        {pieData.map((entry, idx) => (
                          <Cell key={idx} fill={entry.fill} stroke="transparent" />
                        ))}
                      </Pie>
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const d = payload[0].payload;
                          return (
                            <div
                              style={{
                                background: '#161b22',
                                border: '1px solid #30363d',
                                borderRadius: 10,
                                padding: '8px 12px',
                                boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
                              }}
                            >
                              <p style={{ color: '#e6edf3', fontWeight: 600, fontSize: 12 }}>
                                {d.name}
                              </p>
                              <p style={{ color: '#67e8f9', fontSize: 11 }}>
                                Score: {d.score?.toFixed(4)} ({(d.value * 100).toFixed(1)}%)
                              </p>
                            </div>
                          );
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Candidate pills (collapsible) */}
                <p className="chart-section-label">Candidates</p>
                <div className="candidate-list">
                  {result.chart.map((point) => {
                    const isExpanded = expandedCandidates[point.chunk_id];
                    const isSelected = selectedCandidate?.chunk_id === point.chunk_id;
                    return (
                      <div key={point.chunk_id}>
                        <button
                          type="button"
                          className={`candidate-pill ${isSelected ? 'active' : ''}`}
                          onClick={() => {
                            handleCandidateSelect(point);
                            toggleCandidate(point.chunk_id);
                          }}
                        >
                          <div>
                            <span className="candidate-pill-label">{point.label}</span>
                            <span className="candidate-pill-file" style={{ marginLeft: 8 }}>
                              {point.file_name}
                            </span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span className="candidate-pill-score">{point.score.toFixed(4)}</span>
                            {isExpanded ? (
                              <ChevronDown size={14} style={{ color: '#8b949e' }} />
                            ) : (
                              <ChevronRight size={14} style={{ color: '#8b949e' }} />
                            )}
                          </div>
                        </button>

                        {isExpanded && (
                          <div className="candidate-detail">
                            <div className="candidate-detail-header">
                              <span className="candidate-detail-label">Paragraph Preview</span>
                              <span className="candidate-detail-meta">
                                {point.label} • {point.score.toFixed(4)}
                              </span>
                            </div>
                            <p className="candidate-detail-text">{point.text}</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="right-empty">
                <BarChart3 size={40} />
                <p>Select a PYQ to view score charts and candidate evidence.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
