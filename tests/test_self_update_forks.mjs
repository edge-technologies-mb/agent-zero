import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

function loadStore() {
  const code = fs.readFileSync(new URL("../webui/components/settings/external/self-update-store.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "")
    .replace("export { store };", "globalThis.store = store;");
  const calls = [];
  const context = vm.createContext({
    createStore: (_name, model) => model,
    API: { callJsonApi: async (endpoint) => { calls.push(endpoint); return context.response; } },
    console,
  });
  vm.runInContext(code, context);
  return { store: context.store, calls, context };
}

const forkInfo = {
  success: true,
  repository: { url: "https://github.com/example/agent-zero.git", is_custom: true },
  current: { short_tag: "v2.15", branch: "main" },
  defaults: { branch: "main", tag: "latest" },
  available_tag_options: [{ value: "latest", label: "latest (v2.15+1)" }],
};

test("fork quick mode cannot schedule an official release", async () => {
  const { store, calls } = loadStore();
  store.info = forkInfo;
  assert.equal(store.isCustomRepository, true);
  assert.equal(store.quickUpdateAvailable, false);
  assert.equal(store.quickStatusLabel, "CUSTOM REPOSITORY");
  assert.equal(store.repositoryUrl, forkInfo.repository.url);
  await store.scheduleQuickUpdate();
  assert.equal(calls.length, 0);
});

test("refresh selects fork main/latest in Advanced", async () => {
  const { store, context, calls } = loadStore();
  context.response = forkInfo;
  await store.refresh();
  assert.equal(store.error, "");
  assert.equal(store.form.branch, "main");
  assert.equal(store.form.tag, "latest");
  assert.deepEqual(calls, ["self_update_get"]);
});

test("official mode retains manual Advanced version selection", async () => {
  const { store, context } = loadStore();
  context.response = { ...forkInfo, repository: { is_custom: false }, defaults: { branch: "main", tag: "v2.15" } };
  await store.refresh();
  assert.equal(store.isCustomRepository, false);
  assert.equal(store.form.tag, "");
  assert.equal(store.repositoryUrl, "https://github.com/agent0ai/agent-zero.git");
});
