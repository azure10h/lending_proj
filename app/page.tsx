"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Database,
  FileWarning,
  FlaskConical,
  Landmark,
  LineChart as LineIcon,
  Loader2,
  LockKeyhole,
  RefreshCw,
  Scale,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingUp,
  UserRoundSearch,
  XCircle,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Tab = "overview" | "model" | "decision" | "monitoring" | "governance";

type MetricSet = {
  roc_auc: number;
  pr_auc: number;
  ks: number;
  brier: number;
  log_loss: number;
  ece: number;
  approval_rate: number;
  approved_bad_rate: number;
  rejection_capture: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
};

type DashboardData = {
  generatedAt: string;
  portfolio: {
    loans: number;
    chargedOff: number;
    badRate: number;
    fundedAmount: number;
    matureModelPopulation: number;
    dateStart: string;
    dateEnd: string;
  };
  yearly: Array<{ year: number; loans: number; chargedOff: number; amount: number; badRate: number }>;
  segments: {
    grade: Segment[];
    term: Segment[];
    purpose: Segment[];
    homeOwnership: Segment[];
  };
  missingness: Array<{ feature: string; percent: number }>;
  rejected: {
    total: number;
    amount: number;
    riskScoreCoverage: number;
    byYear: Array<{ year: number; applications: number; amount: number; riskPresent: number; riskCoverage: number }>;
    topStates: Array<{ state: string; applications: number }>;
  };
  model: {
    champion: string;
    version: string;
    validated: boolean;
    cutoff: number;
    trainingRows: number;
    validationRows: number;
    testRows: number;
    trainBadRate: number;
    validationBadRate: number;
    testBadRate: number;
    candidates: Record<string, { validation: MetricSet; test: MetricSet }>;
    calibration: Array<{ bucket: number; actual: number; predicted: number; loans: number }>;
    thresholds: Array<{ threshold: number; approvalRate: number; approvedBadRate: number; badCaptured: number }>;
  };
};

type Segment = { label: string; loans: number; chargedOff: number; amount: number; badRate: number };

type NumericRule = {
  impute: number;
  mean: number;
  scale: number;
  p01: number;
  p50: number;
  p99: number;
};

type ModelContract = {
  modelVersion: string;
  champion: string;
  validated: boolean;
  cutoff: number;
  numericFeatures: string[];
  categoricalFeatures: string[];
  numeric: Record<string, NumericRule>;
  categories: Record<string, string[]>;
  calibration: { x: number[]; y: number[] };
  references: { numeric: Record<string, number>; categorical: Record<string, string> };
  reasonLabels: Record<string, string>;
};

type Applicant = Record<string, string | number>;

type ScoreResult = {
  pd: number;
  score: number;
  policySignal: "APPROVE" | "REJECT";
  recommendation: "APPROVE" | "REJECT" | "NOT VALIDATED";
  reasons: string[];
  warnings: string[];
};

const DEFAULT_APPLICANT: Applicant = {
  loan_amnt: 10000,
  annual_inc: 75000,
  dti: 18,
  open_acc: 10,
  pub_rec: 0,
  revol_bal: 12000,
  revol_util: 45,
  total_acc: 24,
  mort_acc: 1,
  pub_rec_bankruptcies: 0,
  credit_history_years: 12,
  term: "36",
  emp_length: "10+ years",
  home_ownership: "MORTGAGE",
  verification_status: "Verified",
  purpose: "debt_consolidation",
  application_type: "INDIVIDUAL",
};

const NAV_ITEMS: Array<{ id: Tab; label: string; icon: typeof BarChart3 }> = [
  { id: "overview", label: "Portfolio Overview", icon: BarChart3 },
  { id: "model", label: "Model Performance", icon: Activity },
  { id: "decision", label: "Decision Simulator", icon: UserRoundSearch },
  { id: "monitoring", label: "Population Monitoring", icon: LineIcon },
  { id: "governance", label: "Data & Governance", icon: ShieldCheck },
];

const GRADE_COLORS = ["#1f8a70", "#4aa786", "#e4a83a", "#dc7a35", "#cf5548", "#a93b50", "#762d42"];

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 });

function percent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function metricPercent(value: number | null | undefined, digits = 1) {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function interpolate(value: number, xs: number[], ys: number[]) {
  if (value <= xs[0]) return ys[0];
  if (value >= xs[xs.length - 1]) return ys[ys.length - 1];
  let low = 0;
  let high = xs.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (xs[middle] <= value) low = middle;
    else high = middle;
  }
  const width = xs[high] - xs[low];
  const ratio = width === 0 ? 0 : (value - xs[low]) / width;
  return ys[low] + ratio * (ys[high] - ys[low]);
}

function LoadingScreen() {
  return (
    <main className="loading-screen">
      <div className="loading-mark"><Landmark size={26} /></div>
      <Loader2 className="spin" size={22} />
      <p>Loading underwriting research workspace…</p>
    </main>
  );
}

function StatCard({ label, value, detail, tone = "default" }: { label: string; value: string; detail: string; tone?: "default" | "green" | "amber" }) {
  return (
    <article className={`stat-card stat-${tone}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

function Panel({ title, eyebrow, action, children, className = "" }: { title: string; eyebrow?: string; action?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-head">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h3>{title}</h3>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string | number }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      {payload.map((item) => (
        <span key={item.name}><i style={{ background: item.color }} />{item.name}: {typeof item.value === "number" ? item.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : item.value}</span>
      ))}
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [contract, setContract] = useState<ModelContract | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [applicant, setApplicant] = useState<Applicant>(DEFAULT_APPLICANT);
  const [threshold, setThreshold] = useState(0.15);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [scoring, setScoring] = useState(false);
  const [loadError, setLoadError] = useState("");
  const sessionRef = useRef<any>(null);
  const ortRef = useRef<any>(null);

  useEffect(() => {
    Promise.all([
      fetch("/data/dashboard-data.json").then((response) => response.json()),
      fetch("/model/model-contract.json").then((response) => response.json()),
    ])
      .then(([dashboard, model]) => {
        setData(dashboard);
        setContract(model);
        setThreshold(model.cutoff);
      })
      .catch(() => setLoadError("Dashboard artifacts could not be loaded."));
  }, []);

  const testMetrics = data ? data.model.candidates[data.model.champion].test : null;
  const currentSignal = result ? (result.pd <= threshold ? "APPROVE" : "REJECT") : null;

  const ensureSession = async () => {
    if (sessionRef.current) return sessionRef.current;
    const ort = await import("onnxruntime-web/wasm");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "/wasm/";
    const session = await ort.InferenceSession.create("/model/model.onnx", { executionProviders: ["wasm"] });
    ortRef.current = ort;
    sessionRef.current = session;
    return session;
  };

  const vectorize = (profile: Applicant, collectWarnings = true) => {
    if (!contract) return { vector: [] as number[], warnings: [] as string[] };
    const vector: number[] = [];
    const warnings: string[] = [];
    contract.numericFeatures.forEach((feature) => {
      const rule = contract.numeric[feature];
      let value = Number(profile[feature]);
      if (!Number.isFinite(value)) value = rule.impute;
      if (value < rule.p01 || value > rule.p99) {
        if (collectWarnings) warnings.push(`${feature.replaceAll("_", " ")} is outside the central 98% of training values and was capped.`);
        value = Math.min(rule.p99, Math.max(rule.p01, value));
      }
      vector.push((value - rule.mean) / (rule.scale || 1));
    });
    contract.categoricalFeatures.forEach((feature) => {
      const value = String(profile[feature] ?? "Missing/Unknown");
      const categories = contract.categories[feature];
      if (!categories.includes(value) && collectWarnings) warnings.push(`${feature.replaceAll("_", " ")} was not observed in model training.`);
      categories.forEach((category) => vector.push(value === category ? 1 : 0));
    });
    return { vector, warnings };
  };

  const scoreProfile = async (profile: Applicant, collectWarnings = false) => {
    if (!contract) throw new Error("Model contract unavailable");
    const session = await ensureSession();
    const { vector, warnings } = vectorize(profile, collectWarnings);
    const ort = ortRef.current;
    const output = await session.run({ input: new ort.Tensor("float32", Float32Array.from(vector), [1, vector.length]) });
    const values = Array.from(output.probabilities.data as Float32Array);
    const rawPd = Number(values[1]);
    const pd = interpolate(rawPd, contract.calibration.x, contract.calibration.y);
    return { pd, warnings };
  };

  const handleScore = async () => {
    if (!contract) return;
    setScoring(true);
    setResult(null);
    try {
      const main = await scoreProfile(applicant, true);
      const impacts: Array<{ feature: string; impact: number }> = [];
      for (const feature of [...contract.numericFeatures, ...contract.categoricalFeatures]) {
        const reference = feature in contract.references.numeric
          ? contract.references.numeric[feature]
          : contract.references.categorical[feature];
        if (String(reference) === String(applicant[feature])) continue;
        const comparison = await scoreProfile({ ...applicant, [feature]: reference });
        impacts.push({ feature, impact: main.pd - comparison.pd });
      }
      const reasons = impacts
        .filter((item) => item.impact > 0.001)
        .sort((a, b) => b.impact - a.impact)
        .slice(0, 4)
        .map((item) => contract.reasonLabels[item.feature]);
      const policySignal = main.pd <= threshold ? "APPROVE" : "REJECT";
      setResult({
        pd: main.pd,
        score: Math.round((1 - main.pd) * 100),
        policySignal,
        recommendation: contract.validated ? policySignal : "NOT VALIDATED",
        reasons: reasons.length ? reasons : ["No material adverse factor was identified against the model reference profile."],
        warnings: main.warnings,
      });
    } catch (error) {
      setLoadError(error instanceof Error ? `Scoring failed: ${error.message}` : "Scoring failed.");
    } finally {
      setScoring(false);
    }
  };

  const resetApplicant = () => {
    setApplicant(DEFAULT_APPLICANT);
    setResult(null);
    setThreshold(contract?.cutoff ?? 0.15);
  };

  if (loadError && !data) return <main className="error-screen"><FileWarning size={30} /><h1>Workspace unavailable</h1><p>{loadError}</p></main>;
  if (!data || !contract || !testMetrics) return <LoadingScreen />;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><Landmark size={22} /></div>
          <div><strong>Northstar Risk</strong><span>Underwriting Lab</span></div>
        </div>
        <nav aria-label="Dashboard sections">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={activeTab === item.id ? "nav-item active" : "nav-item"} onClick={() => setActiveTab(item.id)}>
                <Icon size={17} />
                <span>{item.label}</span>
                {item.id === "decision" && <i>NEW</i>}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-note">
          <FlaskConical size={17} />
          <div><strong>Research environment</strong><span>Not for production credit decisions</span></div>
        </div>
        <div className="sidebar-version"><span>Model</span><strong>{data.model.version}</strong></div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <span className="crumb">Lending Club / Credit Risk</span>
            <h1>{NAV_ITEMS.find((item) => item.id === activeTab)?.label}</h1>
          </div>
          <div className="topbar-actions">
            <span className="data-pill"><Database size={14} /> Data through Q4 2018</span>
            <span className={`validation-pill ${data.model.validated ? "valid" : "warning"}`}>
              {data.model.validated ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
              {data.model.validated ? "Validated" : "Validation gates not met"}
            </span>
          </div>
        </header>

        <div className="content">
          {loadError && <div className="inline-alert"><AlertTriangle size={16} />{loadError}</div>}
          {activeTab === "overview" && (
            <>
              <section className="hero-strip">
                <div>
                  <span className="eyebrow">HISTORICAL CREDIT PORTFOLIO</span>
                  <h2>Risk intelligence across <em>{integer.format(data.portfolio.loans)}</em> completed loans.</h2>
                  <p>Track origination quality, segment loss patterns, and the population used to develop the underwriting research model.</p>
                </div>
                <button className="primary-button" onClick={() => setActiveTab("decision")}><UserRoundSearch size={17} /> Score a new application <ChevronRight size={16} /></button>
              </section>

              <section className="stats-grid">
                <StatCard label="Completed loans" value={compact.format(data.portfolio.loans)} detail={`${data.portfolio.dateStart} – ${data.portfolio.dateEnd}`} />
                <StatCard label="Funded principal" value={money.format(data.portfolio.fundedAmount)} detail="Historical originated amount" />
                <StatCard label="Charged off" value={compact.format(data.portfolio.chargedOff)} detail={`${data.portfolio.badRate.toFixed(1)}% observed bad rate`} tone="amber" />
                <StatCard label="Mature model population" value={compact.format(data.portfolio.matureModelPopulation)} detail={`${percent(data.portfolio.matureModelPopulation / data.portfolio.loans)} of completed loans`} tone="green" />
              </section>

              <section className="dashboard-grid two-one">
                <Panel title="Origination volume and observed loss" eyebrow="PORTFOLIO TREND" action={<span className="panel-meta">Annual cohorts</span>}>
                  <div className="chart-large">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={data.yearly} margin={{ top: 16, right: 12, bottom: 0, left: 0 }}>
                        <CartesianGrid stroke="#e7e9e6" vertical={false} />
                        <XAxis dataKey="year" tickLine={false} axisLine={false} tick={{ fill: "#6a706b", fontSize: 12 }} />
                        <YAxis yAxisId="left" tickFormatter={(v) => compact.format(v)} tickLine={false} axisLine={false} tick={{ fill: "#6a706b", fontSize: 11 }} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 30]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} tick={{ fill: "#6a706b", fontSize: 11 }} />
                        <Tooltip content={<ChartTooltip />} />
                        <Bar yAxisId="left" dataKey="loans" name="Loans" fill="#d8e5df" radius={[3, 3, 0, 0]} />
                        <Line yAxisId="right" type="monotone" dataKey="badRate" name="Bad rate %" stroke="#b64a42" strokeWidth={2.5} dot={{ r: 3, fill: "#b64a42" }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="chart-footnote"><AlertTriangle size={13} /> 2016 contains only terminal outcomes and should not be read as a fully seasoned cohort.</p>
                </Panel>

                <Panel title="Risk concentration" eyebrow="QUICK READ">
                  <div className="insight-list">
                    <article><span className="insight-icon danger"><ArrowUpRight size={16} /></span><div><strong>60-month term</strong><p>{data.segments.term.find((x) => x.label === "60")?.badRate.toFixed(1)}% observed bad rate</p></div><em>2.0× 36m</em></article>
                    <article><span className="insight-icon danger"><TrendingUp size={16} /></span><div><strong>Grade G</strong><p>{data.segments.grade.find((x) => x.label === "G")?.badRate.toFixed(1)}% observed bad rate</p></div><em>Highest</em></article>
                    <article><span className="insight-icon safe"><ArrowDownRight size={16} /></span><div><strong>Grade A</strong><p>{data.segments.grade.find((x) => x.label === "A")?.badRate.toFixed(1)}% observed bad rate</p></div><em>Lowest</em></article>
                    <article><span className="insight-icon neutral"><Target size={16} /></span><div><strong>Small business</strong><p>{data.segments.purpose.find((x) => x.label === "small_business")?.badRate.toFixed(1)}% observed bad rate</p></div><em>Purpose watch</em></article>
                  </div>
                </Panel>
              </section>

              <section className="dashboard-grid equal">
                <Panel title="Observed bad rate by grade" eyebrow="RISK TIER">
                  <div className="chart-medium">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={[...data.segments.grade].sort((a, b) => a.label.localeCompare(b.label))} margin={{ top: 10, right: 8, bottom: 0, left: -12 }}>
                        <CartesianGrid stroke="#e7e9e6" vertical={false} />
                        <XAxis dataKey="label" tickLine={false} axisLine={false} />
                        <YAxis tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} />
                        <Tooltip content={<ChartTooltip />} />
                        <Bar dataKey="badRate" name="Bad rate %" radius={[4, 4, 0, 0]}>
                          {[...data.segments.grade].sort((a, b) => a.label.localeCompare(b.label)).map((entry, index) => <Cell key={entry.label} fill={GRADE_COLORS[index]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
                <Panel title="Largest purpose segments" eyebrow="USE OF FUNDS">
                  <div className="segment-table">
                    <div className="table-row table-header"><span>Purpose</span><span>Loans</span><span>Bad rate</span></div>
                    {data.segments.purpose.slice(0, 6).map((row) => (
                      <div className="table-row" key={row.label}><strong>{row.label.replaceAll("_", " ")}</strong><span>{compact.format(row.loans)}</span><span className={row.badRate >= 22 ? "risk-text" : ""}>{row.badRate.toFixed(1)}%</span></div>
                    ))}
                  </div>
                </Panel>
              </section>
            </>
          )}

          {activeTab === "model" && (
            <>
              <section className="model-banner">
                <div className="model-icon"><Sparkles size={23} /></div>
                <div><span className="eyebrow">CHAMPION MODEL</span><h2>{data.model.champion}</h2><p>Calibrated probability of default · out-of-time test population</p></div>
                <div className="model-status"><AlertTriangle size={17} /><div><strong>Research candidate</strong><span>KS and calibration gates narrowly missed</span></div></div>
              </section>
              <section className="stats-grid model-stats">
                <StatCard label="ROC–AUC" value={testMetrics.roc_auc.toFixed(3)} detail="Discrimination on holdout" tone="green" />
                <StatCard label="KS statistic" value={testMetrics.ks.toFixed(3)} detail="Gate: ≥ 0.250" tone="amber" />
                <StatCard label="Calibration error" value={metricPercent(testMetrics.ece, 1)} detail="Gate: ≤ 3.0%" tone="amber" />
                <StatCard label="Approval at 15% PD" value={metricPercent(testMetrics.approval_rate)} detail={`${metricPercent(testMetrics.approved_bad_rate)} historical bad rate`} />
              </section>
              <section className="dashboard-grid equal">
                <Panel title="Calibration by risk decile" eyebrow="PREDICTED VS OBSERVED">
                  <div className="chart-large">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data.model.calibration} margin={{ top: 14, right: 10, left: -4, bottom: 0 }}>
                        <CartesianGrid stroke="#e7e9e6" vertical={false} />
                        <XAxis dataKey="bucket" tickLine={false} axisLine={false} label={{ value: "Risk decile", position: "insideBottomRight", offset: -2, fill: "#777" }} />
                        <YAxis tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} />
                        <Tooltip content={<ChartTooltip />} />
                        <Line dataKey="predicted" name="Predicted PD %" stroke="#1f6d5b" strokeWidth={2.5} dot={{ r: 3 }} />
                        <Line dataKey="actual" name="Observed bad rate %" stroke="#b64a42" strokeWidth={2.5} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
                <Panel title="Decision policy frontier" eyebrow="THRESHOLD ANALYSIS" action={<span className="panel-meta">15% default</span>}>
                  <div className="chart-large">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={data.model.thresholds} margin={{ top: 14, right: 10, left: -4, bottom: 0 }}>
                        <defs><linearGradient id="approval" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2e8770" stopOpacity={0.35} /><stop offset="100%" stopColor="#2e8770" stopOpacity={0.03} /></linearGradient></defs>
                        <CartesianGrid stroke="#e7e9e6" vertical={false} />
                        <XAxis dataKey="threshold" tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} />
                        <YAxis tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} />
                        <Tooltip content={<ChartTooltip />} />
                        <ReferenceLine x={0.15} stroke="#9c6a2d" strokeDasharray="4 4" label={{ value: "Policy", fill: "#9c6a2d", fontSize: 11 }} />
                        <Area type="monotone" dataKey="approvalRate" name="Approval rate %" stroke="#1f6d5b" fill="url(#approval)" strokeWidth={2.2} />
                        <Line type="monotone" dataKey="approvedBadRate" name="Approved bad rate %" stroke="#b64a42" strokeWidth={2.2} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
              </section>
              <Panel title="Champion–challenger comparison" eyebrow="MODEL SELECTION">
                <div className="model-table-wrap">
                  <table className="model-table"><thead><tr><th>Model</th><th>Role</th><th>Test AUC</th><th>Test KS</th><th>Brier</th><th>ECE</th><th>15% approval</th></tr></thead><tbody>
                    {Object.entries(data.model.candidates).map(([name, candidate]) => <tr key={name}><td><strong>{name}</strong>{name === data.model.champion && <span className="champion-tag">Champion</span>}</td><td>{name === "Logistic Regression" ? "Interpretable baseline" : "Performance challenger"}</td><td>{candidate.test.roc_auc.toFixed(3)}</td><td>{candidate.test.ks.toFixed(3)}</td><td>{candidate.test.brier.toFixed(3)}</td><td>{metricPercent(candidate.test.ece, 1)}</td><td>{metricPercent(candidate.test.approval_rate)}</td></tr>)}
                  </tbody></table>
                </div>
              </Panel>
            </>
          )}

          {activeTab === "decision" && (
            <>
              <section className="decision-intro">
                <div><span className="eyebrow">CLIENT-SIDE MODEL INFERENCE</span><h2>Evaluate a new application</h2><p>Applicant data stays in this browser. The result is a research signal, not a production credit decision.</p></div>
                <span className="privacy-badge"><LockKeyhole size={16} /> No applicant data transmitted</span>
              </section>
              <section className="decision-layout">
                <Panel title="Applicant & loan details" eyebrow="APPLICATION INPUT" action={<button className="text-button" onClick={resetApplicant}><RefreshCw size={14} /> Reset</button>} className="form-panel">
                  <div className="form-section-title"><span>01</span><div><strong>Loan request</strong><p>Requested terms and use of funds</p></div></div>
                  <div className="form-grid">
                    <NumberField label="Requested amount" prefix="$" value={Number(applicant.loan_amnt)} min={500} max={40000} step={500} onChange={(v) => setApplicant({ ...applicant, loan_amnt: v })} />
                    <SelectField label="Term" value={String(applicant.term)} options={[{ value: "36", label: "36 months" }, { value: "60", label: "60 months" }]} onChange={(v) => setApplicant({ ...applicant, term: v })} />
                    <SelectField label="Loan purpose" value={String(applicant.purpose)} options={contract.categories.purpose.map((v) => ({ value: v, label: v.replaceAll("_", " ") }))} onChange={(v) => setApplicant({ ...applicant, purpose: v })} />
                    <SelectField label="Application type" value={String(applicant.application_type)} options={contract.categories.application_type.map((v) => ({ value: v, label: v }))} onChange={(v) => setApplicant({ ...applicant, application_type: v })} />
                  </div>
                  <div className="form-section-title"><span>02</span><div><strong>Income & stability</strong><p>Applicant capacity and employment profile</p></div></div>
                  <div className="form-grid">
                    <NumberField label="Annual income" prefix="$" value={Number(applicant.annual_inc)} min={0} step={1000} onChange={(v) => setApplicant({ ...applicant, annual_inc: v })} />
                    <NumberField label="Debt-to-income" suffix="%" value={Number(applicant.dti)} min={0} max={100} step={0.1} onChange={(v) => setApplicant({ ...applicant, dti: v })} />
                    <SelectField label="Employment length" value={String(applicant.emp_length)} options={contract.categories.emp_length.map((v) => ({ value: v, label: v }))} onChange={(v) => setApplicant({ ...applicant, emp_length: v })} />
                    <SelectField label="Home ownership" value={String(applicant.home_ownership)} options={contract.categories.home_ownership.map((v) => ({ value: v, label: v }))} onChange={(v) => setApplicant({ ...applicant, home_ownership: v })} />
                    <SelectField label="Income verification" value={String(applicant.verification_status)} options={contract.categories.verification_status.map((v) => ({ value: v, label: v }))} onChange={(v) => setApplicant({ ...applicant, verification_status: v })} />
                    <NumberField label="Credit history" suffix="years" value={Number(applicant.credit_history_years)} min={0} max={70} step={1} onChange={(v) => setApplicant({ ...applicant, credit_history_years: v })} />
                  </div>
                  <div className="form-section-title"><span>03</span><div><strong>Credit profile</strong><p>Accounts, utilization, and public records</p></div></div>
                  <div className="form-grid compact-fields">
                    <NumberField label="Open accounts" value={Number(applicant.open_acc)} min={0} step={1} onChange={(v) => setApplicant({ ...applicant, open_acc: v })} />
                    <NumberField label="Total accounts" value={Number(applicant.total_acc)} min={0} step={1} onChange={(v) => setApplicant({ ...applicant, total_acc: v })} />
                    <NumberField label="Revolving balance" prefix="$" value={Number(applicant.revol_bal)} min={0} step={500} onChange={(v) => setApplicant({ ...applicant, revol_bal: v })} />
                    <NumberField label="Revolving utilization" suffix="%" value={Number(applicant.revol_util)} min={0} max={150} step={0.1} onChange={(v) => setApplicant({ ...applicant, revol_util: v })} />
                    <NumberField label="Mortgage accounts" value={Number(applicant.mort_acc)} min={0} step={1} onChange={(v) => setApplicant({ ...applicant, mort_acc: v })} />
                    <NumberField label="Public records" value={Number(applicant.pub_rec)} min={0} step={1} onChange={(v) => setApplicant({ ...applicant, pub_rec: v })} />
                    <NumberField label="Bankruptcies" value={Number(applicant.pub_rec_bankruptcies)} min={0} step={1} onChange={(v) => setApplicant({ ...applicant, pub_rec_bankruptcies: v })} />
                  </div>
                  <button className="score-button" onClick={handleScore} disabled={scoring}>{scoring ? <Loader2 className="spin" size={18} /> : <CircleGauge size={18} />}{scoring ? "Running calibrated model…" : "Generate risk assessment"}<ChevronRight size={17} /></button>
                </Panel>

                <div className="result-column">
                  {!result ? (
                    <section className="result-placeholder"><div className="placeholder-rings"><CircleGauge size={34} /></div><h3>Assessment ready when you are</h3><p>Complete the application fields and run the model to see calibrated PD, a research policy signal, and applicant-specific risk factors.</p><div className="placeholder-steps"><span><i>1</i> Enter application</span><span><i>2</i> Run model</span><span><i>3</i> Review signal</span></div></section>
                  ) : (
                    <>
                      <section className={`decision-card ${currentSignal === "APPROVE" ? "approve" : "reject"}`}>
                        <div className="decision-card-head"><span>RESEARCH POLICY SIGNAL</span><span className="model-chip">{contract.champion}</span></div>
                        <div className="decision-main">
                          <div className="score-gauge" style={{ "--score": `${result.score * 3.6}deg` } as React.CSSProperties}><div><strong>{result.score}</strong><span>Risk score</span></div></div>
                          <div className="recommendation"><span>{currentSignal === "APPROVE" ? <CheckCircle2 size={18} /> : <XCircle size={18} />}{currentSignal}</span><strong>{percent(result.pd)}</strong><p>Calibrated probability of default</p></div>
                        </div>
                        <div className="validation-warning"><AlertTriangle size={16} /><div><strong>Recommendation: NOT VALIDATED</strong><p>The holdout KS and calibration gates were not met. Do not use this signal for a real lending decision.</p></div></div>
                      </section>
                      <Panel title="Decision threshold" eyebrow="SCENARIO POLICY">
                        <div className="threshold-control"><div><strong>{Math.round(threshold * 100)}% PD</strong><span>Approve at or below this level</span></div><input aria-label="Probability of default threshold" type="range" min="5" max="30" value={Math.round(threshold * 100)} onChange={(event) => setThreshold(Number(event.target.value) / 100)} /></div>
                        <div className="threshold-labels"><span>Conservative 5%</span><span>Permissive 30%</span></div>
                      </Panel>
                      <Panel title="Principal adverse factors" eyebrow="LOCAL SENSITIVITY">
                        <ol className="reason-list">{result.reasons.map((reason, index) => <li key={reason}><span>{index + 1}</span><p>{reason}</p></li>)}</ol>
                      </Panel>
                      {result.warnings.length > 0 && <Panel title="Input confidence warnings" eyebrow="OUT-OF-DISTRIBUTION"><ul className="warning-list">{result.warnings.map((warning) => <li key={warning}><AlertTriangle size={14} />{warning}</li>)}</ul></Panel>}
                    </>
                  )}
                </div>
              </section>
            </>
          )}

          {activeTab === "monitoring" && (
            <>
              <section className="stats-grid">
                <StatCard label="Rejected applications" value={compact.format(data.rejected.total)} detail="2007–2018 application population" />
                <StatCard label="Requested principal" value={money.format(data.rejected.amount)} detail="Rejected application demand" />
                <StatCard label="Risk score coverage" value={percent(data.rejected.riskScoreCoverage)} detail={`${percent(1 - data.rejected.riskScoreCoverage)} missing`} tone="amber" />
                <StatCard label="Model feature population" value={compact.format(data.portfolio.matureModelPopulation)} detail="Observed completed outcomes" tone="green" />
              </section>
              <section className="dashboard-grid two-one">
                <Panel title="Rejected application growth" eyebrow="APPLICATION MARKET">
                  <div className="chart-large">
                    <ResponsiveContainer width="100%" height="100%"><AreaChart data={data.rejected.byYear} margin={{ top: 12, right: 10, left: 2 }}><defs><linearGradient id="rejected" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#345f88" stopOpacity={0.35} /><stop offset="100%" stopColor="#345f88" stopOpacity={0.03} /></linearGradient></defs><CartesianGrid stroke="#e7e9e6" vertical={false} /><XAxis dataKey="year" tickLine={false} axisLine={false} /><YAxis tickFormatter={(v) => compact.format(v)} tickLine={false} axisLine={false} /><Tooltip content={<ChartTooltip />} /><Area dataKey="applications" name="Rejected applications" stroke="#345f88" strokeWidth={2.4} fill="url(#rejected)" /></AreaChart></ResponsiveContainer>
                  </div>
                </Panel>
                <Panel title="Data coverage alert" eyebrow="RISK SCORE">
                  <div className="coverage-ring" style={{ "--coverage": `${data.rejected.riskScoreCoverage * 360}deg` } as React.CSSProperties}><div><strong>{percent(data.rejected.riskScoreCoverage, 0)}</strong><span>available</span></div></div>
                  <p className="coverage-copy">Rejected-loan <code>Risk_Score</code> is missing for most applications and is not used as a default outcome.</p>
                </Panel>
              </section>
              <section className="dashboard-grid equal">
                <Panel title="Risk score coverage by year" eyebrow="SOURCE QUALITY">
                  <div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.rejected.byYear} margin={{ top: 10, left: -10 }}><CartesianGrid stroke="#e7e9e6" vertical={false} /><XAxis dataKey="year" tickLine={false} axisLine={false} /><YAxis tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="riskCoverage" name="Coverage %" fill="#d29a3d" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div>
                </Panel>
                <Panel title="Largest rejected markets" eyebrow="STATE DISTRIBUTION">
                  <div className="state-list">{data.rejected.topStates.slice(0, 8).map((row, index) => <div key={row.state}><span><i>{index + 1}</i><strong>{row.state}</strong></span><div><b style={{ width: `${(row.applications / data.rejected.topStates[0].applications) * 100}%` }} /></div><em>{compact.format(row.applications)}</em></div>)}</div>
                </Panel>
              </section>
            </>
          )}

          {activeTab === "governance" && (
            <>
              <section className="governance-hero"><div><Scale size={25} /><span className="eyebrow">MODEL RISK MANAGEMENT</span><h2>Transparent by design. Limited by the data.</h2><p>The workspace separates historical evidence, model output, and decision policy so each can be reviewed independently.</p></div><a href="https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm" target="_blank" rel="noreferrer">Federal model risk guidance <ChevronRight size={14} /></a></section>
              <section className="governance-grid">
                <Panel title="Included model features" eyebrow="APPLICATION-TIME DATA"><div className="feature-chips">{[...contract.numericFeatures, ...contract.categoricalFeatures].map((feature) => <span key={feature}><CheckCircle2 size={13} />{feature.replaceAll("_", " ")}</span>)}</div></Panel>
                <Panel title="Excluded from underwriting" eyebrow="LEAKAGE & PROXY CONTROL"><div className="excluded-list">{["Loan status (target)", "Grade and sub-grade", "Interest rate and installment", "Raw address, ZIP and state", "Employment title and free text", "Initial listing status"].map((item) => <span key={item}><XCircle size={14} />{item}</span>)}</div></Panel>
              </section>
              <section className="dashboard-grid equal">
                <Panel title="Known limitations" eyebrow="REQUIRED CONTEXT">
                  <div className="limitation-list">
                    <article><span>01</span><div><strong>No reject outcomes</strong><p>Declined applicants have no repayment performance, so sample-selection bias remains.</p></div></article>
                    <article><span>02</span><div><strong>No protected-class validation</strong><p>Race, sex, and age are absent. Fair-lending validation cannot be completed from this dataset.</p></div></article>
                    <article><span>03</span><div><strong>Historical population</strong><p>The observation period ends in 2018 and does not represent current credit or economic conditions.</p></div></article>
                    <article><span>04</span><div><strong>Performance gates missed</strong><p>Out-of-time KS was {testMetrics.ks.toFixed(3)} and calibration error was {metricPercent(testMetrics.ece, 1)}.</p></div></article>
                  </div>
                </Panel>
                <Panel title="Model lifecycle controls" eyebrow="BEFORE PRODUCTION USE">
                  <div className="control-timeline">{[
                    ["Independent validation", "Reproduce data, code, calibration, and policy tests"],
                    ["Fair-lending review", "Obtain protected attributes under controlled governance"],
                    ["Current-data redevelopment", "Train on institutional applications and observed outcomes"],
                    ["Ongoing monitoring", "Track drift, discrimination, calibration, overrides, and outcomes"],
                  ].map(([title, copy], index) => <div key={title}><span>{index + 1}</span><article><strong>{title}</strong><p>{copy}</p></article></div>)}</div>
                </Panel>
              </section>
              <Panel title="Source data quality" eyebrow="MISSINGNESS IN MODEL FEATURES"><div className="missing-grid">{data.missingness.slice(0, 6).map((item) => <div key={item.feature}><span><strong>{item.feature.replaceAll("_", " ")}</strong><em>{item.percent.toFixed(2)}%</em></span><div><i style={{ width: `${Math.min(100, item.percent * 8)}%` }} /></div></div>)}</div></Panel>
            </>
          )}
        </div>
      </div>
    </main>
  );
}

function NumberField({ label, value, onChange, prefix, suffix, min, max, step = 1 }: { label: string; value: number; onChange: (value: number) => void; prefix?: string; suffix?: string; min?: number; max?: number; step?: number }) {
  return <label className="field"><span>{label}</span><div className="input-wrap">{prefix && <i>{prefix}</i>}<input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />{suffix && <em>{suffix}</em>}</div></label>;
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return <label className="field"><span>{label}</span><div className="select-wrap"><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select><ChevronRight size={15} /></div></label>;
}
