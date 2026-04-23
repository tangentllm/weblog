import { promises as fs } from "node:fs";
import path from "node:path";

const postsDir = path.resolve("content/posts");
const manifestPath = path.resolve("content/posts/manifest.json");

function parseFrontmatter(fileContent) {
  const match = fileContent.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) {
    return null;
  }
  const lines = match[1].split(/\r?\n/);
  const data = {};
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || !line.includes(":")) {
      continue;
    }
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
  const files = await fs.readdir(postsDir);
  const markdownFiles = files.filter((file) => file.endsWith(".md")).sort();

  const manifest = [];
  for (const fileName of markdownFiles) {
    const fullPath = path.join(postsDir, fileName);
    const content = await fs.readFile(fullPath, "utf-8");
    const meta = parseFrontmatter(content);
    if (!meta) {
      continue;
    }
    const slug = meta.slug || fileName.replace(/\.md$/, "");
    manifest.push({
      slug,
      title: meta.title || slug,
      date: meta.date || "2026-01-01",
      readTime: meta.readTime || "8 分钟",
      category: meta.category || "随想与思考",
      tags: parseTags(meta.tags),
      cover: meta.cover || `https://picsum.photos/seed/${slug}/1000/500`,
      excerpt: meta.excerpt || "",
      file: `./content/posts/${fileName}`,
    });
  }

  manifest.sort((a, b) => new Date(b.date) - new Date(a.date));
  await fs.writeFile(
    manifestPath,
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf-8",
  );
  console.log(`Manifest generated: ${manifest.length} posts`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
