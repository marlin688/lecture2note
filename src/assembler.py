from datetime import datetime


def assemble_markdown(notes: dict) -> str:
    """将结构化笔记 JSON 组装为 Markdown 文档。"""
    lines = []

    title = notes.get("title", "未命名笔记")
    subject = notes.get("subject", "")
    summary = notes.get("summary", "")
    sections = notes.get("sections", [])
    key_terms = notes.get("key_terms", [])
    review_questions = notes.get("review_questions", [])

    # 1. 标题
    lines.append(f"# {title}")
    lines.append("")

    # 2. 元信息
    date_str = datetime.now().strftime("%Y-%m-%d")
    meta_parts = []
    if subject:
        meta_parts.append(f"**学科**：{subject}")
    meta_parts.append(f"**日期**：{date_str}")
    meta_parts.append("**来源**：课堂录音转写")
    lines.append(" | ".join(meta_parts))
    lines.append("")

    # 3. 课程概要
    if summary:
        lines.append("## 📋 课程概要")
        lines.append("")
        for line in summary.split("\n"):
            lines.append(f"> {line}")
        lines.append("")

    # 4. 目录（超过2个section时生成）
    if len(sections) > 2:
        lines.append("## 📑 目录")
        lines.append("")
        for i, sec in enumerate(sections, 1):
            heading = sec.get("heading", f"第 {i} 节")
            anchor = f"section-{i}"
            lines.append(f"{i}. [{heading}](#{anchor})")
        lines.append("")

    # 5. 正文
    for i, sec in enumerate(sections, 1):
        heading = sec.get("heading", f"第 {i} 节")
        content = sec.get("content", "")
        key_points = sec.get("key_points", [])
        emphasis = sec.get("teacher_emphasis")

        anchor = f"section-{i}"
        lines.append(f'## <a id="{anchor}"></a>{i}. {heading}')
        lines.append("")

        # 可折叠的本节要点
        if key_points:
            lines.append("<details>")
            lines.append("<summary>本节要点</summary>")
            lines.append("")
            for point in key_points:
                lines.append(f"- {point}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # 正文内容
        if content:
            lines.append(content)
            lines.append("")

        # 老师强调
        if emphasis:
            lines.append(f"> ⚠️ **老师强调**：{emphasis}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # 6. 术语表
    if key_terms:
        lines.append("## 📖 术语表")
        lines.append("")
        lines.append("| 术语 | 定义 |")
        lines.append("|------|------|")
        for term in key_terms:
            t = term.get("term", "").replace("|", "\\|")
            d = term.get("definition", "").replace("|", "\\|")
            lines.append(f"| {t} | {d} |")
        lines.append("")

    # 7. 复习自测
    if review_questions:
        lines.append("## ✅ 复习自测")
        lines.append("")
        for i, q in enumerate(review_questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    # 8. 页脚
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("---")
    lines.append("")
    lines.append(f"*本笔记由 AI 自动生成于 {gen_time}，仅供学习参考，请以课堂实际内容为准。*")
    lines.append("")

    return "\n".join(lines)
