// Verify that every literal t('...') key in src has a zh_CN translation and
// that placeholders match. Exit 1 on missing keys or placeholder mismatches;
// unused dictionary keys are reported as warnings.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const src = join(dirname(fileURLToPath(import.meta.url)), '..', 'src');
const dict = {};
let conflicts = 0;
const dictDir = join(src, 'i18n', 'zh_CN');
for (const name of readdirSync(dictDir).filter((n) => n.endsWith('.json')).sort()) {
  const part = JSON.parse(readFileSync(join(dictDir, name), 'utf8'));
  for (const [key, value] of Object.entries(part)) {
    if (key in dict && dict[key] !== value) {
      console.error(`conflicting zh_CN for ${JSON.stringify(key)} in ${name}: ${JSON.stringify(dict[key])} vs ${JSON.stringify(value)}`);
      conflicts++;
    }
    dict[key] = value;
  }
}
const files = [];
(function walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path);
    else if (/\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name) && !name.endsWith('.d.ts') && relative(src, path) !== join('i18n', 'index.ts')) files.push(path);
  }
})(src);

// t('literal') / t("literal") / t(`literal`) (no interpolation) and tk('literal') markers.
const pattern = /\bt[k]?\(\s*(['"`])((?:\\.|(?!\1).)*?)\1\s*[,)]/gs;
const used = new Map();
for (const file of files) {
  const text = readFileSync(file, 'utf8');
  for (const match of text.matchAll(pattern)) {
    if (match[1] === '`' && match[2].includes('${')) continue;
    const key = match[2].replace(/\\(['"`\\])/g, '$1').replace(/\\n/g, '\n');
    if (!used.has(key)) used.set(key, relative(src, file));
  }
}
const placeholders = (s) => [...s.matchAll(/\{(\d+)\}/g)].map((m) => m[1]).sort().join(',');
let failed = conflicts;
for (const [key, file] of used) {
  if (!(key in dict)) {
    console.error(`missing zh_CN: ${JSON.stringify(key)}  (${file})`);
    failed++;
  } else if (placeholders(key) !== placeholders(dict[key])) {
    console.error(`placeholder mismatch: ${JSON.stringify(key)} -> ${JSON.stringify(dict[key])}`);
    failed++;
  }
}
const unused = Object.keys(dict).filter((k) => !used.has(k) && !k.startsWith('@'));
if (unused.length) console.warn(`warning: ${unused.length} unused zh_CN keys (dynamic keys are expected), e.g. ${unused.slice(0, 5).map((k) => JSON.stringify(k)).join(', ')}`);
console.log(`i18n: ${used.size} keys used, ${failed} problems`);
process.exit(failed ? 1 : 0);
