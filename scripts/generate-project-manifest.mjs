import { promises as fs } from "node:fs";
import path from "node:path";

const projectsDir = path.resolve("content/projects");
const manifestPath = path.resolve("content/projects/manifest.json");

function parseFrontmatter(fileContent) {
  const match = fileContent.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) return null;
  const lines = match[1].split(/\r?\n/);
  const data = {};
  for (const lineRaw of lines) {
    const line = lineRaw.trim();
    if (!line || !line.includes(":")) continue;
    const [key, ...rest] = line.split(":");
    data[key.trim()] = rest.join(":").trim();
  }
  return data;
}

function parseTags(tagText = "") {
  return tagText
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

async function main() {
  const files = await fs.readdir(projectsDir);
  const markdownFiles = files.filter((file) => file.endsWith(".md")).sort();
  const manifest = [];

  for (const fileName of markdownFiles) {
    const fullPath = path.join(projectsDir, fileName);
    const content = await fs.readFile(fullPath, "utf-8");
    const meta = parseFrontmatter(content);
    if (!meta) continue;

    const slug = meta.slug || fileName.replace(/\.md$/, "");
    manifest.push({
      slug,
      title: meta.title || slug,
      subtitle: meta.subtitle || "",
      status: meta.status || "进行中",
      period: meta.period || "2026",
      tags: parseTags(meta.tags),
      summary: meta.summary || "",
      cover: meta.cover || `https://picsum.photos/seed/${slug}/1200/700`,
      architecture: [],
      features: [],
      metrics: [],
      screenshots: [],
      links: {
        github: meta.github || "#",
        demo: meta.demo || "#",
        docs: meta.docs || "#",
      },
      file: `./content/projects/${fileName}`,
    });
  }

  manifest.sort((a, b) => String(b.period).localeCompare(String(a.period)));
  await fs.writeFile(
    manifestPath,
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf-8",
  );
  console.log(`Project manifest generated: ${manifest.length} projects`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
