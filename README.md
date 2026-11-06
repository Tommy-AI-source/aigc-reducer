# AIGC 降重审计助手

基于 Streamlit 和 DeepSeek API 开发的 AIGC 文本降重与审计工具。

## 核心功能
*   **三轮降重协议**：通过结构化改写、注入人类主观语态、学术语体校准，有效降低 AI 痕迹。
*   **效果可视化**：前端实时展示初始 AIGC 率、修改后 AIGC 率与模拟查重率，为用户提供直观反馈。
*   **技术边界探索**：真实测试了市面商用检测工具（如 PaperPass），摸清了 Prompt 对抗算法的技术天花板。

## 技术栈
*   前端：Streamlit
*   模型：DeepSeek API
*   核心逻辑：Prompt Engineering
