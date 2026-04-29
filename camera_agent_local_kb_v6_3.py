import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


AGENT_SYSTEM_PROMPT = """
你是一个“纯本地知识库版”的相机推荐助手，面向摄影小白。
你的任务不是把用户硬塞进固定表单，而是先理解用户真正想要什么，再决定是否追问，再利用本地知识库完成推荐。

# 可用工具
- understand_user_need(user_message="")
  作用：调用 API 理解用户需求，并把结果写入共享状态。若不传参数，默认基于当前已知的全部用户信息进行理解。
- ask_user_clarification(question="")
  作用：向用户进一步追问一个具体问题。一次只问一个最关键的问题。用户回答会自动写入共享状态。
- search_cameras_by_need()
  作用：基于共享状态中的最新需求理解结果，在本地知识库中初筛并排序候选机型。
- get_camera_details(camera_name="")
  作用：读取某一台相机的详细信息与评分。
- compare_cameras(camera_names="")
  作用：对比多台相机。camera_names 用英文逗号分隔。

# 工作原则
1. 先理解需求，再决定是否需要追问；不要一上来就机械追问。
2. 只有在信息真的影响推荐结论时，才使用 ask_user_clarification。
3. 推荐理由必须适合摄影小白理解。
4. 严禁假装联网，严禁编造本地知识库里不存在的信息。
5. 如果某台机型没有完整评分，要明确说“这台主要依据规格判断”。
6. 最终回答尽量包含：最推荐哪一台、理由、谁不适合选它、备选是谁。
7. 不要把大段 JSON 塞进 Action 参数；需求理解结果已经由工具写入共享状态。

# 输出格式要求
你的每次回复必须严格遵循以下格式，且只输出一对 Thought 和 Action：

Thought: [你的思考和下一步计划]
Action: [具体行动]

Action 只能是以下两类之一：
1. function_name(arg_name="arg_value") 或 function_name()
2. Finish[最终答案]

# 额外要求
- 每次只输出一对 Thought-Action。
- Action 必须在同一行。
- 当你已经有足够依据时，必须 Finish，不要无限追问或无限查工具。
- 如果需要追问，一次只问一个最关键的问题。
"""


NEED_UNDERSTANDING_SYSTEM_PROMPT = """
你是“相机购买需求理解器”。
你的任务是把用户的自然语言需求，转成一个简洁但有用的需求理解摘要，供推荐 Agent 继续行动。

重要前提：
1. 文中提到的预算默认仅指机身预算，不包括镜头、存储卡、电池、三脚架等配件。
2. 你只能围绕本地知识库里已有的机型来理解和建议，不要联想到知识库之外的机型。

你必须只输出 JSON。
JSON 格式如下：
{
  "explicit_needs": ["..."],
  "implicit_preferences": ["..."],
  "missing_info": ["..."],
  "search_focus": ["..."],
  "budget_hint": "...或null",
  "recommended_next_step": "search 或 ask",
  "enough_for_first_recommendation": true
}

要求：
1. explicit_needs 只写用户明确表达过的内容。
2. implicit_preferences 可以写基于用户表达做出的弱推断，但必须保守，最多 3 条。
3. missing_info 只写真正会影响推荐结果的缺失信息。
4. search_focus 写检索时该重点看的维度，例如“拍照分、对焦、重量、机身防抖、视频分、屏幕、续航、镜头生态、麦克风接口”。
5. budget_hint 如果用户没说预算，就填 null；如果用户提到预算，要按“仅机身预算”理解。
6. recommended_next_step 只能是 search 或 ask。
7. 如果已有信息足以先给候选范围，就优先写 search，而不是过度追问。
8. 不要假设用户会额外购买高价镜头来弥补机身短板，除非用户自己明确提到。
9. enough_for_first_recommendation 表示当前信息是否已足以给出第一轮候选推荐。
"""


SEARCH_RANKING_SYSTEM_PROMPT = """
你是“本地相机知识库排序器”。
你的任务是根据用户需求摘要，对候选相机做初筛和排序。

重要前提：
1. 如果用户提到预算，默认这是机身预算，不包括镜头和其他配件。
2. 你只能在提供的本地知识库候选中排序，不能推荐知识库外的机型。

你必须只输出 JSON。
JSON 格式如下：
{
  "ranked_names": ["机型1", "机型2", "机型3"],
  "reasons": {
    "机型1": "一句话原因",
    "机型2": "一句话原因"
  },
  "ruled_out": ["如果有明显不匹配的机型，可写在这里"]
}

要求：
1. 只能根据提供的候选摘要判断，不能使用外部知识。
2. ranked_names 最多返回 5 台，按推荐优先级排序。
3. 如果用户明确提到预算，请优先考虑预算，并按“机身预算”理解；没有明确预算时，不要因为价格高就直接排除所有高价机型，但可以说明性价比风险。
4. 推荐理由要尽量贴近摄影小白的真实使用场景。
5. 如果候选中某机型评分缺失，但规格明显匹配，也可以入选，同时原因里要说明“主要依据规格判断”。
"""


KB_PATH = Path(__file__).with_name("local_camera_kb_beginner_v2.json")


def load_camera_kb() -> List[Dict[str, Any]]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["cameras"]


CAMERA_KB = load_camera_kb()
AGENT_STATE: Dict[str, Any] = {
    "user_context": [],
    "latest_need_profile": None,
    "last_ranking": None,
}


def _camera_name_list_text() -> str:
    names = [camera["name"] for camera in CAMERA_KB]
    return "、".join(names)


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str, silent: bool = False) -> str:
        if not silent:
            print("正在调用大语言模型...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content or ""
            if not silent:
                print("大语言模型响应成功。")
            return answer
        except Exception as e:
            if not silent:
                print(f"调用 LLM API 时发生错误: {e}")
            return f"错误：调用语言模型服务时出错。详细信息：{e}"

    # 【新增】真正的 JSON 输出调用：API 层要求 json_object，并支持一次重试。
    def generate_json(self, prompt: str, system_prompt: str, silent: bool = False, retry: int = 1) -> Tuple[Optional[Dict[str, Any]], str]:
        last_raw = ""
        for attempt in range(retry + 1):
            try:
                if not silent:
                    print("正在调用结构化输出模型...")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    response_format={"type": "json_object"},
                )
                last_raw = response.choices[0].message.content or ""
                parsed = _extract_json_object(last_raw)
                if isinstance(parsed, dict):
                    return parsed, last_raw
                prompt = (
                    prompt
                    + "\n\n你上一次没有返回合法 JSON。请这一次只返回单个 JSON 对象，不要带任何解释。"
                )
            except Exception as e:
                last_raw = f"错误：调用结构化输出时出错。详细信息：{e}"
                if attempt >= retry:
                    return None, last_raw
        return None, last_raw


llm = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _camera_price(camera: Dict[str, Any]) -> int:
    return int(camera.get("price", {}).get("min_cny") or 999999)


def _format_price(camera: Dict[str, Any]) -> str:
    p = camera.get("price", {})
    return p.get("display") or str(p.get("min_cny") or "未知")


def _camera_summary_for_ranking(camera: Dict[str, Any]) -> str:
    photo = camera.get("photo_scores") or {}
    video = camera.get("video_scores") or {}
    stabilization = camera.get("stabilization") or {}
    ports = camera.get("ports") or {}
    screen = camera.get("screen") or {}
    notes = []
    if not camera.get("score_coverage", {}).get("photo_scored"):
        notes.append("拍照评分缺失")
    if not camera.get("score_coverage", {}).get("video_scored"):
        notes.append("视频评分缺失")

    return (
        f"机型：{camera['name']}\n"
        f"价格：约{_format_price(camera)}元；档位：{camera.get('price_band')}\n"
        f"重量：{camera.get('weight_g', '未知')}g；发布日期：{camera.get('release_date') or '未知'}\n"
        f"拍照：综合{photo.get('overall', '暂无')} / 画质{photo.get('image_quality', '暂无')} / 对焦{photo.get('autofocus', '暂无')} / 防抖{photo.get('stabilization', '暂无')} / 续航{photo.get('battery', '暂无')}\n"
        f"视频：综合{video.get('overall', '暂无')} / 对焦{video.get('autofocus', '暂无')} / 规格{video.get('recording_spec', '暂无')} / 连续录制{video.get('continuous_recording', '暂无')} / 音频{video.get('audio', '暂无')}\n"
        f"机身防抖：{stabilization.get('type', '未知')}；CIPA：{stabilization.get('cipa_stops', '未知')}档\n"
        f"屏幕：{screen.get('type', '未知')} {screen.get('size_inch', '未知')}英寸；连拍：{camera.get('burst_fps', '未知')}张/秒\n"
        f"接口：麦克风{'有' if ports.get('mic_jack') else '无'} / 耳机{'有' if ports.get('headphone_jack') else '无'} / USB充电{'支持' if ports.get('usb_charging') else '不支持'}\n"
        f"小白标签：{'、'.join(camera.get('beginner_tags') or ['无'])}\n"
        f"备注：{camera.get('notes') or '无'}"
        + (f"\n说明：{'；'.join(notes)}" if notes else "")
    )


def _combined_user_context(extra_user_message: str = "") -> str:
    parts = list(AGENT_STATE.get("user_context") or [])
    extra = (extra_user_message or "").strip()
    if extra:
        parts.append(extra)
    return "\n".join(parts).strip()


# 【新增】最小 schema 校验：不合法就触发重试或报错，而不是直接吞掉。
def _validate_need_profile(data: Dict[str, Any]) -> Tuple[bool, str]:
    required_list_fields = ["explicit_needs", "implicit_preferences", "missing_info", "search_focus"]
    for field in required_list_fields:
        if field not in data or not isinstance(data[field], list):
            return False, f"字段 {field} 必须是 list"
    if data.get("budget_hint", None) is not None and not isinstance(data.get("budget_hint"), str):
        return False, "字段 budget_hint 必须是字符串或 null"
    if data.get("recommended_next_step") not in {"search", "ask"}:
        return False, "字段 recommended_next_step 只能是 search 或 ask"
    if not isinstance(data.get("enough_for_first_recommendation"), bool):
        return False, "字段 enough_for_first_recommendation 必须是布尔值"
    return True, ""


def _validate_ranking_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(data.get("ranked_names"), list):
        return False, "字段 ranked_names 必须是 list"
    if not data.get("ranked_names"):
        return False, "字段 ranked_names 不能为空"
    if not isinstance(data.get("reasons"), dict):
        return False, "字段 reasons 必须是 dict"
    if "ruled_out" in data and not isinstance(data.get("ruled_out"), list):
        return False, "字段 ruled_out 必须是 list"
    kb_names = {c["name"] for c in CAMERA_KB}
    valid_count = sum(1 for name in data.get("ranked_names", []) if name in kb_names)
    if valid_count == 0:
        return False, "ranked_names 中没有知识库内合法机型"
    return True, ""


# 【新增】统一的 JSON 工具调用：结构化输出 + schema 校验 + 一次重试。
def _call_json_tool(prompt: str, system_prompt: str, validator, tool_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if llm is None:
        return None, "错误：LLM 客户端未初始化。"

    raw_history: List[str] = []
    current_prompt = prompt
    for attempt in range(2):
        parsed, raw = llm.generate_json(current_prompt, system_prompt=system_prompt, silent=True, retry=0)
        raw_history.append(raw)
        if isinstance(parsed, dict):
            ok, err = validator(parsed)
            if ok:
                return parsed, raw
            current_prompt = (
                prompt
                + f"\n\n你刚才返回的 JSON 有问题：{err}。请重新输出一个完全合法的 JSON 对象，不要带任何解释。"
            )
            continue
        current_prompt = (
            prompt
            + "\n\n你刚才没有返回可解析的 JSON。请重新输出一个完全合法的 JSON 对象，不要带任何解释。"
        )
    return None, " | ".join(raw_history)


# 【修改】需求理解结果写入共享状态；后续工具不再接收整段 JSON 字符串。
def understand_user_need(user_message: str = "") -> str:
    context_text = _combined_user_context(user_message)
    if not context_text:
        return "错误：当前没有可供理解的用户需求。"

    prompt = (
        "请理解下面这段用户需求，并输出规定 JSON。\n"
        "提醒：这里的预算默认仅指机身预算，不含镜头和其他配件。\n"
        f"本地知识库已有机型：{_camera_name_list_text()}\n\n"
        f"当前全部用户信息：\n{context_text}"
    )
    parsed, raw = _call_json_tool(
        prompt=prompt,
        system_prompt=NEED_UNDERSTANDING_SYSTEM_PROMPT,
        validator=_validate_need_profile,
        tool_name="understand_user_need",
    )
    if not parsed:
        return f"错误：需求理解器未返回合法结果。原始输出：{raw}"

    AGENT_STATE["latest_need_profile"] = parsed
    result_text = [
        "需求理解完成，结果已写入共享状态 latest_need_profile。",
        json.dumps(parsed, ensure_ascii=False, indent=2),
    ]
    return "\n".join(result_text)


def ask_user_clarification(question: str = "") -> str:
    question = (question or "").strip()
    if not question:
        return "错误：问题不能为空。"
    print(f"Agent 追问：{question}")
    answer = input("> ").strip()
    if not answer:
        return "用户没有补充信息。"
    AGENT_STATE.setdefault("user_context", []).append(f"用户补充：{answer}")
    return f"用户补充已写入共享状态：{answer}"


# 【修改】排序器只读取共享状态中的 latest_need_profile，不再吃 need_summary 长字符串参数。
def search_cameras_by_need() -> str:
    need_profile = AGENT_STATE.get("latest_need_profile")
    if not isinstance(need_profile, dict):
        return "错误：当前没有 latest_need_profile。请先调用 understand_user_need()。"

    # 【修改】程序侧初筛只按预算做价格窗口过滤：允许略超，但尽量避免明显低于预算的机型混入。
    candidate_pool = sorted(CAMERA_KB, key=_camera_price)
    budget_hint = str(need_profile.get("budget_hint") or "").strip()
    budget_match = re.search(r"(\d{4,6})", budget_hint)
    if budget_match:
        budget = int(budget_match.group(1))

        # 允许价格略超预算，同时设置一个下限，减少“明显低于预算”的机型进入候选池。
        upper_bound = int(budget * 1.15)
        lower_bound = int(budget * 0.80)

        filtered = [
            c for c in candidate_pool
            if lower_bound <= _camera_price(c) <= upper_bound
        ]

        # 如果窗口太窄导致候选过少，则退回到只限制上限，保证至少有基本可选项。
        if len(filtered) >= 3:
            candidate_pool = filtered
        else:
            fallback = [c for c in candidate_pool if _camera_price(c) <= upper_bound]
            if fallback:
                candidate_pool = fallback

    candidate_summaries = [_camera_summary_for_ranking(camera) for camera in candidate_pool[:10]]
    # input("输入查看candidate_summaries")
    # print(candidate_summaries)
    # input()
    prompt = (
        "以下是共享状态中的最新需求理解结果 JSON：\n"
        f"{json.dumps(need_profile, ensure_ascii=False, indent=2)}\n\n"
        "重要提醒：如果用户提到预算，默认这是机身预算，不包括镜头和其他配件。\n"
        f"本地知识库已有机型：{_camera_name_list_text()}\n\n"
        "以下是本地知识库中的候选机型摘要：\n\n"
        + "\n\n--------------------\n\n".join(candidate_summaries)
        + "\n\n请严格只在上述机型中输出排序 JSON。"
    )
    parsed, raw = _call_json_tool(
        prompt=prompt,
        system_prompt=SEARCH_RANKING_SYSTEM_PROMPT,
        validator=_validate_ranking_result,
        tool_name="search_cameras_by_need",
    )
    if not parsed:
        return f"错误：排序器未返回合法结果。原始输出：{raw}"

    ranked_names = parsed.get("ranked_names") or []
    reasons = parsed.get("reasons") or {}
    ruled_out = parsed.get("ruled_out") or []

    valid_names = []
    for name in ranked_names:
        if any(c["name"] == str(name) for c in candidate_pool):
            valid_names.append(str(name))

    if not valid_names:
        return f"错误：排序器返回的机型都不在候选池中。原始输出：{raw}"

    AGENT_STATE["last_ranking"] = {
        "ranked_names": valid_names,
        "reasons": reasons,
        "ruled_out": ruled_out,
    }

    lines = ["本地知识库按自然语言需求完成的候选排序："]
    for idx, name in enumerate(valid_names[:5], start=1):
        camera = next(c for c in CAMERA_KB if c["name"] == name)
        lines.append(
            f"{idx}. {camera['name']} | 约 {_format_price(camera)} 元 | 重量 {camera.get('weight_g', '未知')}g | "
            f"拍照 {camera.get('photo_scores', {}).get('overall', '暂无')} | 视频 {camera.get('video_scores', {}).get('overall', '暂无')}"
        )
        reason = reasons.get(name) or "未提供简短原因。"
        lines.append(f"   入选理由：{reason}")
        tags = camera.get("beginner_tags") or []
        if tags:
            lines.append(f"   小白提示：{'、'.join(tags)}")
        if not camera.get("score_coverage", {}).get("photo_scored") or not camera.get("score_coverage", {}).get("video_scored"):
            lines.append("   说明：这台部分评分缺失，主要依据规格判断。")

    if ruled_out:
        lines.append("可能不太匹配的方向：" + "；".join(str(x) for x in ruled_out))
    return "\n".join(lines)


def get_camera_details(camera_name: str = "") -> str:
    name = (camera_name or "").strip().lower()
    for c in CAMERA_KB:
        if c["name"].lower() == name:
            photo = c.get("photo_scores") or {}
            video = c.get("video_scores") or {}
            ports = c.get("ports") or {}
            stabilization = c.get("stabilization") or {}
            screen = c.get("screen") or {}
            score_note = []
            if not c.get("score_coverage", {}).get("photo_scored"):
                score_note.append("拍照评分缺失，主要依据规格判断")
            if not c.get("score_coverage", {}).get("video_scored"):
                score_note.append("视频评分缺失，主要依据规格判断")
            return (
                f"机型：{c['name']}\n"
                f"价格：约 {_format_price(c)} 元（档位：{c.get('price_band')}）\n"
                f"发布日期：{c.get('release_date') or '未知'}\n"
                f"卡口：{c.get('mount') or '未知'}\n"
                f"重量：{c.get('weight_g') or '未知'}g\n"
                f"对焦点：{c.get('autofocus_points') or '未知'}\n"
                f"屏幕：{screen.get('type', '未知')}，{screen.get('size_inch', '未知')} 英寸\n"
                f"连拍：{c.get('burst_fps') or '未知'} 张/秒\n"
                f"视频规格：{c.get('video_spec') or '未知'}\n"
                f"机身防抖：{stabilization.get('type', '未知')}"
                + (f"，约 {stabilization.get('cipa_stops')} 档" if stabilization.get('cipa_stops') else "")
                + "\n"
                f"续航：{c.get('battery_life_cipa') or '未知'} 张（CIPA）\n"
                f"接口：麦克风{'有' if ports.get('mic_jack') else '无'} / 耳机{'有' if ports.get('headphone_jack') else '无'} / USB充电{'支持' if ports.get('usb_charging') else '不支持'}\n"
                f"拍照评分：综合 {photo.get('overall', '暂无')}，画质 {photo.get('image_quality', '暂无')}，对焦 {photo.get('autofocus', '暂无')}，防抖 {photo.get('stabilization', '暂无')}，操控 {photo.get('handling', '暂无')}\n"
                f"视频评分：综合 {video.get('overall', '暂无')}，对焦 {video.get('autofocus', '暂无')}，规格 {video.get('recording_spec', '暂无')}，连续录制 {video.get('continuous_recording', '暂无')}，防抖 {video.get('stabilization', '暂无')}，音频 {video.get('audio', '暂无')}\n"
                f"小白标签：{'、'.join(c.get('beginner_tags') or ['无'])}\n"
                f"备注：{c.get('notes') or '无'}"
                + (f"\n说明：{'；'.join(score_note)}" if score_note else "")
            )
    return f"错误：本地知识库中不存在机型 '{camera_name}'"


def compare_cameras(camera_names: str = "") -> str:
    names = [n.strip() for n in camera_names.split(",") if n.strip()]
    if len(names) < 2:
        return "错误：compare_cameras 至少需要 2 台机型，名称之间请用英文逗号分隔。"

    selected = []
    for name in names:
        matched = next((c for c in CAMERA_KB if c["name"].lower() == name.lower()), None)
        if not matched:
            return f"错误：本地知识库中不存在机型 '{name}'"
        selected.append(matched)

    lines = ["机型对比结果："]
    for c in selected:
        photo = c.get("photo_scores") or {}
        video = c.get("video_scores") or {}
        stabilization = c.get("stabilization") or {}
        lines.append(
            f"- {c['name']}: 价格约 {_format_price(c)} 元，拍照 {photo.get('overall', '暂无')}，视频 {video.get('overall', '暂无')}，"
            f"对焦 {photo.get('autofocus', video.get('autofocus', '暂无'))}，防抖 {stabilization.get('type', '未知')}，"
            f"重量 {c.get('weight_g', '未知')}g，屏幕 {c.get('screen', {}).get('type', '未知')}"
        )
        if not c.get("score_coverage", {}).get("photo_scored") or not c.get("score_coverage", {}).get("video_scored"):
            lines.append("  说明：这台部分评分缺失，主要依据规格判断。")
    return "\n".join(lines)


available_tools = {
    "understand_user_need": understand_user_need,
    "ask_user_clarification": ask_user_clarification,
    "search_cameras_by_need": search_cameras_by_need,
    "get_camera_details": get_camera_details,
    "compare_cameras": compare_cameras,
}


# 【新增】更稳的 Action 解析：用 AST 解析函数调用，避免简单正则误伤。
def _parse_action_call(action_str: str) -> Tuple[Optional[str], Dict[str, str], Optional[str]]:
    action_str = action_str.strip()
    try:
        expr = ast.parse(action_str, mode="eval").body
    except Exception as e:
        return None, {}, f"Action 不是合法的函数调用表达式：{e}"

    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return None, {}, "Action 不是支持的函数调用形式。"
    if expr.args:
        return None, {}, "当前只支持关键字参数，不支持位置参数。"

    kwargs: Dict[str, str] = {}
    for kw in expr.keywords:
        if kw.arg is None:
            return None, {}, "当前不支持 **kwargs。"
        if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
            return None, {}, "当前只支持字符串字面量参数。"
        kwargs[kw.arg] = kw.value.value
    return expr.func.id, kwargs, None


def run_agent(user_prompt: str, max_turns: int = 8) -> str:
    AGENT_STATE["user_context"] = [f"用户原始需求：{user_prompt}"]
    AGENT_STATE["latest_need_profile"] = None
    AGENT_STATE["last_ranking"] = None

    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"送入 Agent 的请求:\n{user_prompt}\n" + "=" * 40)

    for i in range(max_turns):
        print(f"--- 循环 {i + 1} ---\n")
        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误：未能解析到 Action 字段。请确保回复严格遵循 'Thought: ... Action: ...' 格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
            final_answer = finish_match.group(1) if finish_match else "错误：Finish 格式不正确。"
            print(f"任务完成，最终答案: {final_answer}")
            return final_answer

        tool_name, kwargs, parse_error = _parse_action_call(action_str)
        if parse_error or not tool_name:
            observation = f"错误：无法解析 Action -> {action_str}。{parse_error or ''}"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        if tool_name in available_tools:
            try:
                observation = available_tools[tool_name](**kwargs)
            except TypeError as e:
                observation = f"错误：工具调用参数不匹配。详细信息：{e}"
            except Exception as e:
                observation = f"错误：工具执行失败。详细信息：{e}"
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)

    final_answer = "已达到最大循环次数，任务未完成。"
    print(final_answer)
    return final_answer


if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL_ID = os.getenv("MODEL_NAME")

    if not API_KEY or not BASE_URL or not MODEL_ID:
        raise ValueError("请在 .env 中配置 OPENAI_API_KEY、OPENAI_BASE_URL、MODEL_NAME")

    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    user_prompt = input("请输入你的相机需求：").strip() or "我想买一台适合旅行拍照的相机，最好容易上手一点。"
    run_agent(user_prompt)
