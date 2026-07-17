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

test("server-renders the Northstar underwriting workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /Northstar Risk \| Lending Club Underwriting Lab/i);
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
  assert.equal(JSON.parse(dashboard).portfolio.loans, 396030);
  assert.equal(JSON.parse(dashboard).rejected.total, 27648741);
  assert.equal(JSON.parse(contract).featureNames.length, 48);
  assert.match(page, /Generate risk assessment/);
  assert.match(page, /NOT VALIDATED/);
  assert.match(packageJson, /onnxruntime-web/);
  assert.doesNotMatch(`${dashboard}${contract}`, /Michelle Gateway|emp_title|zip code/i);
  await access(new URL("../public/model/model.onnx", import.meta.url));
  await access(new URL("../public/og.png", import.meta.url));
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});
