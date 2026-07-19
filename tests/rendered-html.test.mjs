import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html", host: "localhost" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the LendingClub credit-risk workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /LendingClub Credit Risk Analysis/i);
  assert.match(html, /Loading underwriting research workspace/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("ships model and aggregate artifacts without source records", async () => {
  const [dashboard, contract, page, packageJson] = await Promise.all([
    readFile(new URL("../public/data/dashboard-data.json", import.meta.url), "utf8"),
    readFile(new URL("../public/model/model-contract.json", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  const dashboardData = JSON.parse(dashboard);
  assert.equal(dashboardData.portfolio.loans, 396030);
  assert.equal(dashboardData.rejected.total, 27648741);
  assert.equal(dashboardData.model.rejectInference.method, "Post-stratification inverse-propensity weighting");
  assert.deepEqual(Object.keys(dashboardData.model.rejectInference.scenarios), ["3", "5", "10", "20"]);
  assert.ok(dashboardData.model.rejectInference.matchedRejectedApplications > 1_000_000);
  assert.equal(JSON.parse(contract).featureNames.length, 48);
  assert.match(page, /Generate risk assessment/);
  assert.match(page, /Reject inference sensitivity/);
  assert.match(page, /NOT VALIDATED/);
  assert.match(packageJson, /onnxruntime-web/);
  assert.doesNotMatch(`${dashboard}${contract}`, /Michelle Gateway|emp_title|zip code/i);
  await access(new URL("../public/model/model.onnx", import.meta.url));
  await access(new URL("../public/og.png", import.meta.url));
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});
