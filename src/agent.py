import asyncio
import datetime
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import bey, cartesia, deepgram, google, openai, silero

# from livekit.plugins import tavus
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import health_db
# import products_db
import prompter
import web_search

logger = logging.getLogger("agent")
_bg_tasks: set[asyncio.Task] = set()

load_dotenv(".env.local")

# AGENT_MODEL = "openai/gpt-5.3-chat-latest"
AGENT_MODEL = "google/gemini-3.1-flash-lite-preview"
# PRODUCTS_DB_PATH = Path(__file__).parent.parent / "data" / "products.db"
HEALTH_DB_PATH = health_db.HEALTH_DB_PATH

# Voice settings shared between Cartesia direct plugin and LiveKit Inference TTS.
_TTS_VOICE_ID = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
# Must be a single value from Cartesia's TTSVoiceEmotion enum (capitalised).
_TTS_EMOTION = "Affectionate"
_TTS_SPEED = "normal"

# Direct-plugin model defaults (used only when the corresponding API key is set).
_DIRECT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
_DIRECT_OPENAI_MODEL = "gpt-4o-mini"
_DIRECT_STT_MODEL = "nova-2"


def _build_llm():
    """LLM provider selection by env key. Priority: Google → OpenAI → LiveKit Inference."""
    if os.getenv("GOOGLE_API_KEY"):
        logger.info("LLM: using direct Google plugin (model=%s)", _DIRECT_GEMINI_MODEL)
        return google.LLM(model=_DIRECT_GEMINI_MODEL)
    if os.getenv("OPENAI_API_KEY"):
        logger.info("LLM: using direct OpenAI plugin (model=%s)", _DIRECT_OPENAI_MODEL)
        return openai.LLM(model=_DIRECT_OPENAI_MODEL)
    logger.info("LLM: using LiveKit Inference (model=%s)", AGENT_MODEL)
    return inference.LLM(model=AGENT_MODEL)


def _build_stt():
    """Direct Deepgram plugin if DEEPGRAM_API_KEY is set, else LiveKit Inference."""
    if os.getenv("DEEPGRAM_API_KEY"):
        logger.info("STT: using direct Deepgram plugin (model=%s)", _DIRECT_STT_MODEL)
        return deepgram.STT(model=_DIRECT_STT_MODEL, language="th")
    logger.info("STT: using LiveKit Inference (deepgram/nova-2)")
    return inference.STT(model="deepgram/nova-2", language="th")


def _build_tts():
    """OmniVoice if OMNIVOICE_MODEL is set, direct Cartesia if CARTESIA_API_KEY is set, else LiveKit Inference."""
    if os.getenv("OMNIVOICE_MODEL"):
        from omnivoice_tts import OmniVoiceTTS

        model_path = os.getenv("OMNIVOICE_MODEL")
        logger.info("TTS: using OmniVoice (%s)", model_path)
        speed_str = os.getenv("OMNIVOICE_SPEED")
        steps_str = os.getenv("OMNIVOICE_STEPS")
        return OmniVoiceTTS(
            model_path=model_path,
            language=os.getenv("OMNIVOICE_LANGUAGE", "Thai"),
            instruct=os.getenv("OMNIVOICE_INSTRUCT") or None,
            ref_audio=os.getenv("OMNIVOICE_REF_AUDIO") or None,
            ref_text=os.getenv("OMNIVOICE_REF_TEXT") or None,
            speed=float(speed_str) if speed_str else None,
            num_step=int(steps_str) if steps_str else 16,
        )
    # if os.getenv("CARTESIA_API_KEY"):
    #     logger.info("TTS: using direct Cartesia plugin (sonic-3)")
    #     return cartesia.TTS(
    #         model="sonic-3",
    #         voice=_TTS_VOICE_ID,
    #         language="th",
    #         emotion=_TTS_EMOTION,
    #         speed=_TTS_SPEED,
        # )
    logger.info("TTS: using LiveKit Inference (cartesia/sonic-3)")
    return inference.TTS(
        model="cartesia/sonic-3",
        voice=_TTS_VOICE_ID,
        extra_kwargs={"emotion": _TTS_EMOTION, "speed": _TTS_SPEED},
    )


# Core persona
_PERSONA = """
You are a helpful voice AI health assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
Always respond in Thai language.
You are female. Always use female polite particles: end sentences with "ค่ะ" or "นะคะ", use "ดิฉัน" or "ฉัน" to refer to yourself, and use "คุณ" when addressing \
the user.
Keep responses concise and conversational. Do NOT use markdown, asterisks, bullet points, code blocks, or emojis — but DO use natural punctuation \
(commas, periods, ellipses, exclamation, question marks) since these control the pacing and intonation of your spoken voice.
You are curious, friendly, and have a sense of humor.
When greeting the user, ask them what they would like help with today.
"""

_SPOKEN_STYLE_INSTRUCTIONS = """
You are speaking, not writing. Make every reply sound like a real person talking, not text being read aloud.

FILLER WORDS — sprinkle naturally (do not overuse): "อืม...", "เอ่อ...", "แบบว่า", "ก็นะ", "เอาเป็นว่า", "นะคะ", "ค่ะ".
Use them where a real person would pause to think, soften a statement, or transition between ideas.

PUNCTUATION CONTROLS YOUR BREATH AND PITCH — the TTS reads these as pacing cues:
- Comma ( , ) = short breath / mini-pause.
- Period ( . ) = full stop, voice drops, longer pause.
- Ellipsis ( ... ) = trailing thought, hesitation, drawn-out word.
- Exclamation ( ! ) = louder, brighter, more energy.
- Question ( ? ) = rising pitch at the end.
Use them deliberately. A response with no commas sounds robotic and out of breath.

PERFORMANCE TAGS — use sparingly (at most one per reply, only when it fits the moment):
- [breath] — soft inhale, good before delivering important information.
- [sighs] — gentle sigh, good for empathy or slight reluctance.
- [laughs] — light laugh, good for warm or playful moments.
- [whisper] ... [/whisper] — softer voice for confidential or intimate lines.
Place tags inside the sentence, e.g. "เอ่อ [breath] ตอนนี้ค่าน้ำตาลของคุณค่อนข้างสูงนะคะ".

SHORT CHUNKS — prefer two or three short sentences over one long one. Short sentences let the voice breathe and feel less rushed.

AVOID: long monotone paragraphs, commas every ten words, robotic listing ("หนึ่ง สอง สาม"), or stacking fillers ("อืม เอ่อ แบบว่า") in a row.
"""

# Number pronunciation
_NUMBER_INSTRUCTIONS = """
Always speak numbers in Thai, never in English. For prices, quantities, and number of days, use natural Thai numeric words (e.g. 1290 → "หนึ่งพันสองร้อยเก้าสิบบาท", 14 days → "สิบสี่วัน"). For digit
sequences such as document number prefixes, codes, IDs, phone numbers, or years read aloud digit by digit, pronounce each digit in Thai
(e.g. "01" → "ศูนย์หนึ่ง", "2026" said as a year → "สองศูนย์สองหก", "0812345678" → "ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปด").
Never say "zero", "one", "two" in English.
"""

# Health records
_HEALTH_INSTRUCTIONS = """

STEP 1 — Say a brief Thai phrase before calling any tool so the user is not left in silence, e.g. "รอสักครู่นะคะ กำลังดึงข้อมูลสุขภาพให้ค่ะ".

STEP 2 — Fetch health data (ONE call, returns instantly):
- No specific person → call get_latest_health_record. Do NOT ask who first.
- Specific person mentioned → call get_health_record with their name or user_id.
- Unsure who → call list_health_users to clarify, then fetch.

STEP 3 — Speak the health metrics right away in Thai:
- Greet the user by name.
- State each metric as ผลปกติ or ผลผิดปกติ with its value and normal range.

STEP 4 — If the result says "ABNORMAL METRICS FOUND", call get_health_recommendations immediately after speaking.
Then speak the recommendations as practical lifestyle tips per abnormal metric, and close with encouragement.
If the result says "All metrics normal", congratulate the user — do NOT call get_health_recommendations.

Never invent health values. Always call the fetch tool fresh — never reuse a result from earlier in this conversation.
"""

# Web search
_WEB_SEARCH_INSTRUCTIONS = """
For general questions, fact-checking, current events, or any topic not covered by the uploaded documents, use the web_search tool.
Always say a brief Thai phrase before calling web_search so the user is not left in silence (e.g. "ขอค้นหาข้อมูลจากอินเทอร์เน็ตสักครู่นะคะ" or "ขอดูข้อมูลก่อนนะคะ"). Say it BEFORE the tool call, not after.
After receiving the search results, summarize the answer concisely and naturally in Thai.
The results include a "today" date header — use it to judge whether individual results are recent or outdated. For time-sensitive topics, prefer newer sources. If a result shows a "[date]" tag, treat older results with appropriate skepticism.
"""

# Product catalog
# _PRODUCT_INSTRUCTIONS = """
# You help customers find products in our shop. When the user asks about products, prices, or what is available, use the search_products or list_all_products tools \
# to get accurate information from the catalog. Never invent products or prices.
# Translate English product names to Thai when speaking, but pass English keywords to the search tool.
# """

# Documents (uploaded PDFs / text files)
# _DOCUMENT_INSTRUCTIONS = """
# Before answering, read documents, you can choose multiple documents to read.
# When the user asks about anything company or work related — such as reports, plans, policies, procedures, announcements, or internal information — call list_documents first to check whether a relevant document exists. If one or more
# documents look relevant, tell the user which ones you found and ask which they want you to read. Only call read_document after the user has chosen. If no documents match, answer from general knowledge or use web_search.
# Also call list_documents if the user explicitly asks about uploaded files, PDFs, or what documents are available.
# After reading a document, answer the user's question grounded in that text — do not invent facts that are not in the document.
# Translate English document titles to Thai when speaking.
# """

AGENT_INSTRUCTIONS = "\n\n".join(
    [
        _PERSONA,
        _SPOKEN_STYLE_INSTRUCTIONS,
        _NUMBER_INSTRUCTIONS,
        _HEALTH_INSTRUCTIONS,
        _WEB_SEARCH_INSTRUCTIONS,
        # _PRODUCT_INSTRUCTIONS,
        # _DOCUMENT_INSTRUCTIONS,
    ]
)


# class _TavusGreenScreen(tavus.AvatarSession):
#     async def start(self, agent_session: AgentSession, room, **kwargs) -> None:
#         _orig = self._api.create_conversation
#
#         async def _with_greenscreen(**kw):
#             props = dict(kw.pop("properties", None) or {})
#             props["apply_greenscreen"] = True
#             return await _orig(properties=props, **kw)
#
#         self._api.create_conversation = _with_greenscreen
#         await super().start(agent_session, room, **kwargs)


def _build_health_summary(record: health_db.HealthRecord) -> tuple[str, str | None]:
    """Return (report_text, search_query_or_None). Fast — no I/O."""
    analysis = record.range_analysis()
    concerns: list[str] = []
    for line in analysis.splitlines():
        if ": NORMAL" not in line:
            metric = line.split(":")[0].strip()
            label = (
                line.split(": ", 1)[1].split(" (")[0].strip() if ": " in line else ""
            )
            if metric and label:
                concerns.append(f"{metric} {label}")

    report = f"HEALTH RECORD:\n{record.to_summary()}\n\nRANGE ANALYSIS:\n{analysis}"
    query = (
        ("health lifestyle recommendations for " + ", ".join(concerns))
        if concerns
        else None
    )
    return report, query


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=AGENT_INSTRUCTIONS)

    # @function_tool
    # async def search_products(self, context: RunContext, query: str) -> str:
    #     """Search the product catalog by name or description keyword.

    #     Use this when the user asks about a specific product, type of product,
    #     or feature. Pass English keywords (e.g. "headphones", "bluetooth", "cable").

    #     Args:
    #         query: English keyword to search for in product name or description.
    #     """
    #     logger.info(f"Searching products for: {query}")
    #     results = products_db.search_products(PRODUCTS_DB_PATH, query)
    #     if not results:
    #         return f"No products found matching '{query}'."
    #     return "\n".join(p.to_summary() for p in results)

    # @function_tool
    # async def list_all_products(self, context: RunContext) -> str:
    #     """List every product in the catalog.

    #     Use this when the user asks what is available, what you sell, or wants
    #     to browse the full catalog.
    #     """
    #     logger.info("Listing all products")
    #     products = products_db.list_all_products(PRODUCTS_DB_PATH)
    #     return "\n".join(p.to_summary() for p in products)

    # @function_tool
    # async def list_documents(self, context: RunContext) -> str:
    #     """List the uploaded documents (PDFs, text files) in the knowledge base.

    #     Use this when the user asks "what documents do we have?", "what files are
    #     uploaded?", or asks any question about uploaded files. Always call this
    #     first and present the names back to the user before calling read_document.
    #     """
    #     logger.info("Listing backend documents")
    #     docs = backend_docs.list_ready_documents()
    #     if not docs:
    #         return "No documents have been uploaded yet."
    #     return "\n".join(d.to_summary() for d in docs)

    # @function_tool
    # async def read_document(self, context: RunContext, document_id: str) -> str:
    #     """Read the text content of a specific uploaded document.

    #     Only call this AFTER list_documents and AFTER the user has chosen
    #     which document to read. The content is truncated for long files;
    #     the truncation note is included so you can tell the user.

    #     Args:
    #         document_id: The id from list_documents (e.g. "f1c69f7a-..."),
    #             or a partial-id prefix that uniquely identifies the document.
    #     """
    #     logger.info(f"Reading document: {document_id}")
    #     doc = backend_docs.get_document(document_id)
    #     if doc is None:
    #         for candidate in backend_docs.list_ready_documents():
    #             if candidate.id.startswith(document_id):
    #                 doc = candidate
    #                 break
    #     if doc is None:
    #         return f"Document with id '{document_id}' not found."
    #     try:
    #         text = backend_docs.read_document_text(doc)
    #     except FileNotFoundError:
    #         return f"The file for '{doc.name}' is missing on disk."
    #     except ValueError as exc:
    #         return str(exc)
    #     if not text:
    #         return f"'{doc.name}' contains no extractable text."
    #     return f"Content of {doc.name}:\n\n{text}"

    @function_tool
    async def get_latest_health_record(self, context: RunContext) -> str:
        """Return the most-recently-updated health record across ALL users.

        Use when the user asks for "the latest health" without naming a specific person.
        Call immediately — do NOT ask who or list users first. Always call fresh.
        After this returns, speak the health data, then call get_health_recommendations
        if the result says there are abnormal metrics.
        """
        logger.info("Fetching most-recent health record across all users")
        record = await asyncio.to_thread(
            health_db.get_most_recent_health, HEALTH_DB_PATH
        )
        if record is None:
            return "No health records found in the database."
        report, query = _build_health_summary(record)
        if query:
            logger.info(f"Starting background health search: {query}")
            context.session.userdata["_health_search"] = asyncio.create_task(
                web_search.get_default_provider().search(query, max_results=3)
            )
            return (
                report
                + "\n\nABNORMAL METRICS FOUND: call get_health_recommendations next."
            )
        return report + "\n\nAll metrics normal."

    @function_tool
    async def list_health_users(self, context: RunContext) -> str:
        """List all users who have health records in the database.

        Call this only when the user asks who has records, or when you need to
        clarify which person they mean before fetching a specific record.
        """
        logger.info("Listing health users")
        users = await asyncio.to_thread(health_db.list_users, HEALTH_DB_PATH)
        if not users:
            return "No health records found in the database."
        lines = [
            f"{u['name']} (user_id: {u['user_id']}, last updated: {u['last_updated']})"
            for u in users
        ]
        return "\n".join(lines)

    @function_tool
    async def get_health_record(
        self,
        context: RunContext,
        name: str = "",
        user_id: str = "",
    ) -> str:
        """Retrieve the most recent health record for a specific user.

        Looks up by user_id first; falls back to name search.
        After this returns, speak the health data, then call get_health_recommendations
        if the result says there are abnormal metrics.
        Always call fresh — never reuse a prior result.

        Args:
            name: The person's name (Thai or English). Use if user_id is unknown.
            user_id: The exact user_id string (e.g. "user_001"). Preferred over name.
        """
        logger.info(f"Fetching health record — user_id={user_id!r} name={name!r}")
        record = await health_db.get_latest_health_async(
            user_id=user_id or None,
            name=name or None,
            db_path=HEALTH_DB_PATH,
        )
        if record is None:
            return f"No health record found for user_id={user_id!r} / name={name!r}."
        report, query = _build_health_summary(record)
        if query:
            logger.info(f"Starting background health search: {query}")
            context.session.userdata["_health_search"] = asyncio.create_task(
                web_search.get_default_provider().search(query, max_results=3)
            )
            return (
                report
                + "\n\nABNORMAL METRICS FOUND: call get_health_recommendations next."
            )
        return report + "\n\nAll metrics normal."

    @function_tool
    async def get_health_recommendations(self, context: RunContext) -> str:
        """Retrieve lifestyle recommendations for the abnormal metrics from the last health fetch.

        Call this immediately after speaking the health data when the previous health tool
        result said "ABNORMAL METRICS FOUND". The web search runs in background while you
        speak, so this call is usually near-instant.
        """
        task: asyncio.Task | None = context.session.userdata.pop("_health_search", None)
        if task is None:
            return "No pending health recommendations. Call get_health_record first."
        try:
            results = await task
        except Exception as exc:
            logger.warning("Background health search failed: %s", exc)
            return "Web search failed — please give general advice based on the range analysis."
        if not results:
            return "No web results found."
        return "\n".join(r.to_summary() for r in results)

    @function_tool
    async def web_search(self, context: RunContext, query: str) -> str:
        """Search the internet for general information, current events, or fact-checking.

        Use this for questions that require external knowledge from the internet, like "what is USB-C?",
        "how does noise cancellation work?", or "what is the capital of France?".
        Do not use this for internal company files or uploaded PDFs — use search_documents for those.
        Results include publication dates where available so you can judge freshness.

        Args:
            query: English search query (translate from Thai to English for better search results).
        """
        logger.info(f"Web searching: {query}")
        provider = web_search.get_default_provider()
        results = await provider.search(query, max_results=5)
        if not results:
            return f"No web results found for '{query}'."
        today = datetime.date.today().isoformat()
        header = f"[Today: {today}]\n"
        return header + "\n".join(r.to_summary() for r in results)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # products_db.init_db(PRODUCTS_DB_PATH)
    health_db.init_db(HEALTH_DB_PATH)


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice AI pipeline. Each component picks the direct provider plugin if its
    # API key is set, otherwise falls back to LiveKit Inference. See _build_*.
    session = AgentSession(
        stt=_build_stt(),
        llm=_build_llm(),
        tts=_build_tts(),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        userdata={},
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Add a virtual avatar to the session (skipped in console/fake-job mode).
    # For other providers, see https://docs.livekit.io/agents/models/avatar/
    if not ctx.is_fake_job():
        avatar = bey.AvatarSession(avatar_id=os.getenv("BEY_AVATAR_ID"))
        # To switch to Tavus (with green screen), comment out the line above and use:
        # avatar = _TavusGreenScreen(
        #     replica_id=os.getenv("TAVUS_REPLICA_ID"),
        #     persona_id=os.getenv("TAVUS_PERSONA_ID"),
        # )

        # Start the avatar; fall back to voice-only if the provider rejects (e.g. no credits)
        try:
            await avatar.start(session, room=ctx.room)
        except Exception as exc:
            logger.warning("Avatar unavailable, continuing voice-only: %s", exc)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        # Noise cancellation disabled. To re-enable: uncomment the block below
        # and re-add `room_io` and `ai_coustics` to the imports at the top of
        # the file.
        # room_options=room_io.RoomOptions(
        #     audio_input=room_io.AudioInputOptions(
        #         noise_cancellation=ai_coustics.audio_enhancement(
        #             model=ai_coustics.EnhancerModel.QUAIL_VF_L
        #         ),
        #     ),
        # ),
    )

    # Register the session with the prompter and start the local UI server.
    # Open http://localhost:7860 to feed text directly to the agent's TTS.
    prompter.set_session(session)
    _task = asyncio.create_task(prompter.start())
    _task.add_done_callback(lambda t: _bg_tasks.discard(t))
    _bg_tasks.add(_task)

    # Join the room and connect to the user
    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user warmly in Thai and ask what you can help them with today."
    )


if __name__ == "__main__":
    cli.run_app(server)
