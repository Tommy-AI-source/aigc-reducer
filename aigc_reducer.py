# -*- coding: utf-8 -*-
"""
AIGC 降重审计助手
=================
基于 Streamlit + DeepSeek API，按「三轮降重协议」改写学术文本：

    第一轮 · 减法：句式重组（长句拆短、短句合并、主动/被动交替），
                   清除“不仅……更是……”“折射出”等 AI 高频表达。
    第二轮 · 拟人：注入人类学者的主观连接词与自然的节奏起伏，
                   用更平实的书面词替换空泛大词，
                   目标是降低外部商用检测器的 AIGC 判定率。
    第三轮 · 校准：修正真正的语法与指代问题，守住学术语体，
                   同时输出逐条审计说明，
                   并模拟估算初始 / 修改后 AIGC 率与查重率（Demo 演示值，非真实检测）。

运行（PowerShell）：
    python -m pip install --upgrade streamlit openai
    $env:DEEPSEEK_API_KEY="sk-你的key"
    python -m streamlit run aigc_reducer.py

依赖：streamlit >= 1.31, openai >= 1.30
"""

from __future__ import annotations

import difflib
import os
import re

import streamlit as st
from openai import OpenAI


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BASE_URL = "https://api.deepseek.com"
MODELS = ["deepseek-chat", "deepseek-reasoner"]
TEXT_MARK = "<<<TEXT>>>"
AUDIT_MARK = "<<<AUDIT>>>"
RATE_MARK = "<<<RATES>>>"

SYSTEM_PROMPT = (
    "你是一位中文人文社科领域的写作者，长期写论文，也熟悉人类学者的行文习惯。"
    "你的任务是把可能被 AIGC 检测器标记的学术段落，改写成人类学者自然写出的文字："
    "保持学术感与书面语感，句子长短交替、主动被动穿插，允许适度流露个人判断；"
    "绝不使用方言、网络用语、俚语或过度接地气的口语。"
    "不可动摇的底线：不编造事实，不增删原文观点与论据，"
    "人名、年代、数据、关键术语与原意保持一致，改写后句子通顺、指代清楚、逻辑连贯。"
    "目标是“像人写的、读得顺”，不是“糙”或“怪”。"
)

# 拟人化约束：第二轮注入执行，第三轮负责守住 ---------------------------------
ANTI_DETECT_RULES = """【学术拟人化重构规则（五条同时执行）】

A. 守住学术语体
   - 只用书面语。严禁方言、网络用语、俚语和过度接地气的表达，
     例如“搁”“那拨”“刨”“整”“搞”“说白了”“我觉得吧”一律不用。
   - 可以平实、朴素，但不能油滑，不能写成聊天口吻。

B. 清除 AI 高频表达（硬性禁止）
   - 绝对不用“不仅……更是……”“折射出”“深刻地揭示了”“随着……的发展”；
   - 同时尽量避开“值得注意的是”“毋庸置疑”“综上所述”“由此可见”
     “在某种意义上”“换句话说”等模板化短语；确需表达相同意思时，换一种写法。

C. 句式重组
   - 主动语态与被动语态交替使用，不要让整段停在同一语态上；
   - 长句拆成 2—3 个短句；几个零碎短句也可以合并成一个带插入语的长句
     （如“这一点——至少从现有材料看——并不难解释”），但合并后必须读得通；
   - 句长要有起伏，避免连续几句长度相近。

D. 注入人类学者的主观思考
   - 在段落开头或转折处，适度加入“在我看来”“事实上”“令人遗憾的是”
     “客观上来说”“不难看出”“这一点值得追问”等主观连接词；
   - 每段一到两处即可，位置不要固定，必须与上下文自然衔接，不能变成口头禅；
   - 主观词只调节语气与节奏，不改变原文的判断和结论。

E. 高维同义替换
   - 把空泛的大词换成更自然、更平实的书面词：
     “折射出”→“反映出”，“深刻地揭示了”→“点明了”，“具有重要意义”→“颇为关键”，
     “构建”→“建立”，“解构”→“拆解”，“彰显”→“体现出”。
   - 概念性术语可按语境作平实的书面改写，如“精神胜利法”→“自我麻痹的机制”；
     但首次出现或关键论证处应保留原术语，以免概念漂移。
   - 术语、人名、年代、数据必须与原文事实一致，不得编造。

底线（优先级最高）：原意、论点、论据、叙述顺序不变；
改写后必须语法成立、指代清楚、逻辑连贯、读起来自然。
任何一条规则与底线冲突时，以底线为准。"""

# 第一轮：减法（句式重组 + 去 AI 腔） -----------------------------------------
PROMPT_ROUND_1 = """【第一轮 · 减法：句式重组与去 AI 腔】

请改写下面这段学术文本：
1. 长句拆成 2—3 个短句；零碎的短句可以合并成一个带插入语的长句；
2. 主动句与被动句交替使用，至少调整两三处语态；
3. 删除“不仅……更是……”“折射出”“深刻地揭示了”“随着……的发展”；
   句首的“因此”“然而”“此外”“综上所述”如非逻辑必需，直接删去或改写；
4. 把部分空泛大词换成平实的书面词（如“折射出”→“反映出”），
   首次出现的关键术语保留原词；
5. 不增删观点与论据，人名、年代、数据不能改错；
6. 不要加小标题、序号、说明或任何前后缀。

原文：
{text}

只输出改写后的正文。"""

# 第二轮：拟人（学者语气 + 随机节奏） -------------------------------------------
PROMPT_ROUND_2 = """【第二轮 · 拟人：注入学者语气与自然节奏】

在下面这段文字上继续改写，让它读起来像一位人类学者自然写出的段落，
目标是降低外部商用 AIGC 检测器的判定率。
注意：是“自然的学术书面语”，不是口语化，更不是方言或网络用语。

{anti_detect_rules}

执行要求：
1. 按 D 条，在段首或转折处加入一到两处主观连接词（“在我看来”“事实上”
   “令人遗憾的是”“客观上来说”），位置错开，不要每段都放同一处；
2. 按 C 条继续调整句长：该拆的拆，该合的合，主动被动交替；
3. 按 E 条替换剩余的大词，使措辞更平实自然；关键术语首次出现时保留；
4. 通读自检：原意有没有变？逻辑是否连贯？有没有出现方言、网络用语或
   过于口语的表达？有就改掉；
5. 不要形成“每段固定开头 + 固定结尾”的新模板，节奏要自然、不均匀。

文本：
{text}

只输出改写后的正文。"""

# 第三轮：校准 + 审计 + 结果反馈 --------------------------------------------------
PROMPT_ROUND_3 = """【第三轮 · 校准与审计】

{anti_detect_rules}

请完成三件事。

一、校准下面这段文字。这一轮是“守”，不是“重写”：
- 只修真正的语法错误、成分残缺和指代不清（读者会误解的那种），用最少的字补；
- 保留第二轮加入的主观连接词（“在我看来”“事实上”“令人遗憾的是”等），
  只要不密到重复，就原地保留；
- 保留长短句交替、主被动穿插和带插入语的长句，不要改回长度均匀、结构对称的模板腔；
- 清除任何方言、网络用语、俚语或过度口语化的表达；
- 禁止回填“不仅……更是……”“折射出”“深刻地揭示了”“随着……的发展”，
  禁止在段尾补写总结句；
- 核对人名、年代、数据、关键术语与原文是否一致，不一致就改回来；
- 术语替换只要不造成概念漂移，就保留。

二、对照原文，写出改写审计说明。用无序列表列出 5—8 条，覆盖“词”“句”“节奏”三类：
- 词：哪个空泛大词换成了什么更平实的书面词（如“精神胜利法”→“自我麻痹的机制”），
  为什么换；
- 句：哪句拆了、哪句合并了、哪句主被动互换，具体写出来；
- 节奏：哪里加入了主观连接词、哪里形成长短句落差，说明如何增加自然感。
说明要具体到词和句，不要写空泛评价。

三、结果反馈估算：
在输出改写文本和审计说明后，请基于你的经验，模拟估算出原文本的【初始 AIGC 率】、
改写后的【修改后 AIGC 率】以及【模拟查重率】，并严格按照下面的格式输出在最后：
初始 AIGC率: X% | 修改后 AIGC率: Y% | 查重率: Z%
（X / Y / Z 均为 0—100 的整数，不带小数，不加任何解释或补充说明。）

严格按以下格式输出，不要添加其他内容：
{text_mark}
（校准后的最终文本）
{audit_mark}
- ……
- ……
{rate_mark}
初始 AIGC率: X% | 修改后 AIGC率: Y% | 查重率: Z%

待校准文本：
{text}

原文（用于核对、撰写审计与估算初始 AIGC 率，不要照抄）：
{original}
"""


# ---------------------------------------------------------------------------
# 页面与样式
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AIGC降重审计助手", page_icon="🧾", layout="centered")

st.markdown(
    """
    <style>
      :root, html, body, .stApp { color-scheme: light; }
      .stApp { background: #f5f5f2; color: #33302b; }
      .block-container { max-width: 880px; padding-top: 3.2rem; padding-bottom: 4.5rem; }

      /* 标题在浅色背景上必须是深色，避免被深色主题反向覆盖 */
      h1, h2, h3, h4, h5, h6,
      h1 *, h2 *, h3 *,
      [data-testid="stHeading"],
      [data-testid="stHeading"] * {
          color: #1f1d1a !important;
      }
      h1 { font-size: 1.85rem; font-weight: 600; letter-spacing: .02em; }
      p, label, .stMarkdown { color: #33302b; }
      label, [data-testid="stWidgetLabel"] p,
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] p {
          color: #33302b !important;
      }

      /* 输入区强制使用浅色背景 + 深色文字，含所有 textarea 和单行输入框 */
      .stTextArea textarea,
      .stTextInput input,
      [data-testid="stTextArea"] textarea,
      [data-testid="stTextInput"] input,
      textarea,
      input[type="text"],
      input[type="password"] {
          background-color: #ffffff !important;
          color: #333333 !important;
          -webkit-text-fill-color: #333333 !important;
          caret-color: #333333 !important;
          border: 1px solid #ddd9d0 !important;
          border-radius: 8px !important;
          font-size: .95rem;
          line-height: 1.8;
      }
      .stTextArea textarea::placeholder,
      .stTextInput input::placeholder,
      textarea::placeholder,
      input::placeholder {
          color: #8a857a !important;
          -webkit-text-fill-color: #8a857a !important;
          opacity: 1 !important;
      }
      .stTextArea textarea:focus,
      .stTextInput input:focus {
          border-color: #8a857a !important;
          box-shadow: none !important;
      }

      div.stButton > button {
          background: #26241f; color: #f5f5f2; border: 1px solid #26241f;
          border-radius: 8px; padding: .52rem 1.6rem; font-weight: 500;
      }
      div.stButton > button:hover { background: #403c34; color: #fff; border-color: #403c34; }
      section[data-testid="stSidebar"] { background: #efeee9; }
      hr { border-color: #e2ded5; }
      div[data-testid="stMetricValue"] { font-size: 1.9rem; }

      /* 结果反馈指标卡片 */
      .rate-card {
          background: #ffffff;
          border: 1px solid #e3dfd6;
          border-radius: 12px;
          padding: 1rem 1.1rem 0.9rem;
          height: 100%;
          box-shadow: 0 1px 2px rgba(31, 29, 26, .04);
      }
      .rate-card .rate-label {
          font-size: .8rem;
          color: #6f6a60;
          letter-spacing: .04em;
      }
      .rate-card .rate-value {
          font-size: 2rem;
          font-weight: 600;
          line-height: 1.25;
          margin-top: .25rem;
          font-variant-numeric: tabular-nums;
      }
      .rate-card .rate-note {
          font-size: .75rem;
          color: #8a857a;
          margin-top: .3rem;
          min-height: 1.1em;
      }
      .rate-card.red .rate-value { color: #b23a30; }
      .rate-card.green .rate-value { color: #23744a; }
      .rate-card.blue .rate-value { color: #2a5f8f; }

      /* 指标卡片下方的浅灰色免责声明 */
      .rate-disclaimer,
      .rate-disclaimer * {
          color: #a09b90 !important;
          font-size: .76rem;
          line-height: 1.75;
          margin: .6rem 0 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AIGC 降重审计助手")
st.caption("三轮协议：句式重组 → 学术拟人化 → 校准与审计 ｜ 附检测率模拟估算")

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 设置")
    api_key = st.text_input(
        "DeepSeek API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        help="也可先在 PowerShell 中设置环境变量 DEEPSEEK_API_KEY",
    )
    model = st.selectbox("模型", MODELS, index=0)
    st.caption("密钥仅用于当前会话的 API 调用，不会写入任何文件。")
    with st.expander("三轮协议说明"):
        st.markdown(
            "**第一轮 · 减法：句式重组**\n\n"
            "长句拆短、短句合并、主动/被动交替，清除“不仅……更是……”“折射出”"
            "“深刻地揭示了”“随着……的发展”等 AI 高频表达。\n\n"
            "**第二轮 · 拟人：学术语体**\n\n"
            "注入“在我看来”“事实上”“令人遗憾的是”“客观上来说”等学者式主观连接词，"
            "把空泛大词换成平实的书面词，制造自然的长短句节奏；"
            "不口语化，不用方言和网络用语。\n\n"
            "**第三轮 · 校准**\n\n"
            "只修真正的语法与指代问题，守住学术语体，"
            "不把文本改回模板腔，并逐条说明改了什么。"
        )

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def count_chars(text: str) -> int:
    """统计去除空白后的字符数。"""
    return len(re.sub(r"\s+", "", text or ""))


def modification_rate(original: str, final: str) -> float:
    """字符级差异估算修改幅度，返回 0—100 的百分比。"""
    a = re.sub(r"\s+", "", original or "")
    b = re.sub(r"\s+", "", final or "")
    if not a:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return max(0.0, min(100.0, (1.0 - ratio) * 100.0))


def clean_audit(raw: str) -> list[str]:
    """把审计区文本整理成 markdown 列表项。"""
    items: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(\d+[.)、]|[-*•])\s*", "", line)
        if len(line) < 3 or line in {"改写审计说明", "审计说明"}:
            continue
        items.append(f"- {line}")
    return items


# “初始 AIGC率: 78%” / “修改后 AIGC率: 21%” / “模拟查重率: 12%” 一类字段
_RATE_PAIR_RE = re.compile(
    r"([一-龥A-Za-z\s]*?(?:AIGC\s*率|查重率|重复率))\s*[:：=]\s*(\d+(?:\.\d+)?)\s*%?"
)

_EMPTY_RATES = {"initial": None, "final": None, "similarity": None}


def _clamp_pct(value: str) -> float:
    """把字符串百分比转成 0—100 的浮点数。"""
    return max(0.0, min(100.0, float(value)))


def parse_rates(block: str) -> dict[str, float | None]:
    """从模型输出中解析初始 / 修改后 AIGC 率与查重率。

    兼容三种写法：
      * 初始 AIGC率: 78% | 修改后 AIGC率: 21% | 查重率: 12%
      * AIGC率: 21% | 查重率: 12%          （仅两个字段，归入“修改后”）
      * 多行、序号或项目符号开头的同类字段
    """
    rates: dict[str, float | None] = dict(_EMPTY_RATES)
    generic: list[float] = []

    for label, value in _RATE_PAIR_RE.findall(block or ""):
        norm = re.sub(r"\s+", "", label)
        pct = _clamp_pct(value)
        if "查重" in norm or "重复" in norm:
            rates["similarity"] = pct
        elif any(k in norm for k in ("初始", "原文", "改写前", "降重前")):
            rates["initial"] = pct
        elif any(k in norm for k in ("修改后", "改写后", "最终", "降重后")):
            rates["final"] = pct
        else:  # 只有“AIGC率”，通常指改写后的结果
            generic.append(pct)

    if rates["final"] is None and generic:
        rates["final"] = generic[-1]
    return rates


def _is_rate_line(line: str) -> bool:
    """判断某行是否“整行只由率字段和分隔符组成”。"""
    if not _RATE_PAIR_RE.search(line):
        return False
    leftover = _RATE_PAIR_RE.sub("", line)
    leftover = re.sub(r"[\s|｜,，;；、/\\\-–—•*·()（）\[\]【】:：=]+", "", leftover)
    return len(leftover) <= 2


def split_rate_block(raw: str) -> tuple[str, str]:
    """把结果反馈区从正文 / 审计区剥离出来，避免污染审计列表。"""
    if RATE_MARK in raw:
        body, _, rate_part = raw.partition(RATE_MARK)
        return body.rstrip(), rate_part.strip()

    kept: list[str] = []
    rate_lines: list[str] = []
    for line in raw.splitlines():
        if _is_rate_line(line):
            rate_lines.append(line)
        else:
            kept.append(line)
    return "\n".join(kept), "\n".join(rate_lines)


def parse_third_round(raw: str) -> tuple[str, list[str], dict[str, float | None]]:
    """解析第三轮返回的 <<<TEXT>>> / <<<AUDIT>>> / <<<RATES>>> 区块。"""
    raw = (raw or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    raw, rate_block = split_rate_block(raw)
    rates = parse_rates(rate_block)
    if all(v is None for v in rates.values()):
        rates = parse_rates(raw)

    if TEXT_MARK in raw:
        body = raw.split(TEXT_MARK, 1)[1]
        text_part, _, audit_part = body.partition(AUDIT_MARK)
        return text_part.strip(), clean_audit(audit_part), rates
    return raw, [], rates


def fmt_rate(value: float | None) -> str:
    """百分比展示：整数不带小数位，缺失显示为 —。"""
    if value is None:
        return "—"
    return f"{value:.0f}%" if abs(value - round(value)) < 1e-6 else f"{value:.1f}%"


def _rate_card(label: str, value: float | None, tone: str, note: str) -> str:
    return (
        f'<div class="rate-card {tone}">'
        f'<div class="rate-label">{label}</div>'
        f'<div class="rate-value">{fmt_rate(value)}</div>'
        f'<div class="rate-note">{note}</div>'
        f"</div>"
    )


def render_rate_cards(rates: dict[str, float | None]) -> None:
    """渲染三张检测率指标卡片与免责声明。"""
    rates = rates or {}
    initial = rates.get("initial")
    final = rates.get("final")
    similarity = rates.get("similarity")

    if initial is not None and final is not None:
        delta = final - initial
        amount = f"{abs(delta):.1f}".rstrip("0").rstrip(".")
        if delta < 0:
            final_note = f"较初始 ↓{amount} 个百分点"
        elif delta > 0:
            final_note = f"较初始 ↑{amount} 个百分点"
        else:
            final_note = "与初始持平"
    else:
        final_note = "改写后估算"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            _rate_card("初始 AIGC 率", initial, "red", "原文检测基准"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _rate_card("修改后 AIGC 率", final, "green", final_note),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _rate_card("模拟查重率", similarity, "blue", "未接入知网 / 维普"),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="rate-disclaimer">注：本工具为产品演示 Demo，未接入知网/维普数据库。'
        "上述检测率由大模型模拟估算，仅供降重自查参考。</p>",
        unsafe_allow_html=True,
    )


def call_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float = 0.75,
    max_tokens: int = 4096,
) -> str:
    """调用 DeepSeek（OpenAI 兼容接口），返回正文。"""
    kwargs: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "timeout": 180,
    }
    # deepseek-reasoner 不支持 temperature 参数
    if model_name != "deepseek-reasoner":
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# 输入区
# ---------------------------------------------------------------------------
text = st.text_area(
    "待降重文本",
    height=270,
    placeholder="把需要降重的论文段落粘贴到这里，支持长文本，可直接粘贴……",
)

col_a, col_b = st.columns([1, 3])
with col_a:
    run = st.button("开始降重与审计")
with col_b:
    st.caption(f"当前 {count_chars(text)} 字")

if run:
    if not api_key.strip():
        st.error("请先在左侧填写 DeepSeek API Key，或设置环境变量 DEEPSEEK_API_KEY。")

    elif not text.strip():
        st.warning("请先粘贴需要处理的文本。")

    else:
        try:
            client = OpenAI(api_key=api_key.strip(), base_url=BASE_URL)

            with st.status("正在执行三轮降重协议……", expanded=True) as status:
                status.write("第 1 / 3 轮 · 减法：拆分与合并句式，清除 AI 高频表达")
                round1 = call_model(
                    client, model, PROMPT_ROUND_1.format(text=text),
                    temperature=0.7,
                )

                status.write("第 2 / 3 轮 · 拟人：注入学者语气、替换大词、调整节奏")
                round2 = call_model(
                    client,
                    model,
                    PROMPT_ROUND_2.format(
                        text=round1,
                        anti_detect_rules=ANTI_DETECT_RULES,
                    ),
                    temperature=1.0,
                )

                status.write("第 3 / 3 轮 · 校准：修正语法与指代，守住学术语体，生成审计说明")
                round3 = call_model(
                    client,
                    model,
                    PROMPT_ROUND_3.format(
                        text=round2,
                        original=text,
                        anti_detect_rules=ANTI_DETECT_RULES,
                        text_mark=TEXT_MARK,
                        audit_mark=AUDIT_MARK,
                        rate_mark=RATE_MARK,
                    ),
                    temperature=0.4,
                )

                final_text, audit_items, rates = parse_third_round(round3)
                status.update(label="三轮完成", state="complete", expanded=False)

            st.session_state["result"] = {
                "original": text,
                "final": final_text,
                "audit": audit_items,
                "rates": rates,
                "round1": round1,
                "round2": round2,
                "raw3": round3,
            }

        except Exception as exc:  # noqa: BLE001 — 面向用户展示错误即可
            st.error(f"调用失败：{type(exc).__name__}: {exc}")
            st.caption("常见原因：API Key 无效、余额不足、网络超时，或模型名不可用。")

# ---------------------------------------------------------------------------
# 输出区
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res and res.get("final"):
    st.divider()

    # ① 结果反馈指标
    st.subheader("① 检测率估算")
    render_rate_cards(res.get("rates") or {})
    st.write("")

    # ② 改写后文本
    st.subheader("② 改写后文本")
    st.text_area(
        "最终文本",
        value=res["final"],
        height=340,
        label_visibility="collapsed",
    )
    st.download_button(
        "下载为 TXT",
        data=res["final"].encode("utf-8"),
        file_name="aigc_reduced.txt",
        mime="text/plain",
    )

    # ③ 修改率估算
    st.subheader("③ 修改率估算")
    rate = modification_rate(res["original"], res["final"])
    m1, m2, m3 = st.columns(3)
    m1.metric("修改率", f"{rate:.1f}%")
    m2.metric("原文", f"{count_chars(res['original'])} 字")
    m3.metric("改写后", f"{count_chars(res['final'])} 字")
    st.progress(min(rate / 100.0, 1.0))
    st.caption(
        "算法：去除空白字符后做字符级序列比对（difflib.SequenceMatcher），"
        "修改率 = 1 − 相似度。该数值只反映改动幅度，不等同于查重率或任何 AIGC 检测分数。"
    )

    # ④ 改写审计说明
    st.subheader("④ 改写审计说明")
    if res["audit"]:
        st.markdown("\n".join(res["audit"]))
    else:
        st.warning("模型没有按约定格式返回审计列表，以下是第三轮的原始输出：")
        st.text(res.get("raw3", ""))

    with st.expander("查看三轮中间稿"):
        st.markdown("**第一轮 · 减法后**")
        st.text_area("round1", value=res["round1"], height=180, label_visibility="collapsed")
        st.markdown("**第二轮 · 降 AI 后**")
        st.text_area("round2", value=res["round2"], height=180, label_visibility="collapsed")
