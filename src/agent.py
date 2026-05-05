import asyncio
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
from livekit.plugins import silero, tavus
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import backend_docs

# import products_db
import prompter
import web_search

logger = logging.getLogger("agent")
_bg_tasks: set[asyncio.Task] = set()

load_dotenv(".env.local")

AGENT_MODEL = "openai/gpt-5.3-chat-latest"
# PRODUCTS_DB_PATH = Path(__file__).parent.parent / "data" / "products.db"

# Core persona
_PERSONA = """
You are a helpful voice AI shop assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
Always respond in Thai language.
You are female. Always use female polite particles: end sentences with "ค่ะ" or "นะคะ", use "ดิฉัน" or "หนู" to refer to yourself, and use "คุณ" when addressing \
the user.
Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
You are curious, friendly, and have a sense of humor.
When greeting the user, ask them what they would like help with today.
"""

# Product catalog
# _PRODUCT_INSTRUCTIONS = """
# You help customers find products in our shop. When the user asks about products, prices, or what is available, use the search_products or list_all_products tools \
# to get accurate information from the catalog. Never invent products or prices.
# Translate English product names to Thai when speaking, but pass English keywords to the search tool.
# """

# Documents (uploaded PDFs / text files)
_DOCUMENT_INSTRUCTIONS = """
Before answering, read documents, you can choose multiple documents to read.
When the user asks about anything company or work related — such as reports, plans, policies, procedures, announcements, or internal information — call list_documents first to check whether a relevant document exists. If one or more
documents look relevant, tell the user which ones you found and ask which they want you to read. Only call read_document after the user has chosen. If no documents match, answer from general knowledge or use web_search.
Also call list_documents if the user explicitly asks about uploaded files, PDFs, or what documents are available.
After reading a document, answer the user's question grounded in that text — do not invent facts that are not in the document.
Translate English document titles to Thai when speaking.
"""

# Web search
_WEB_SEARCH_INSTRUCTIONS = """
For general questions about a technology, what a product type does, how something works, or any topic outside our shop catalog, use the web_search tool.
Do not use web_search for items already in our catalog — use search_products for those.
Web search takes a couple of seconds, so when you decide to use it, briefly tell the user you are looking it up (e.g. "ขอดูข้อมูลก่อนนะคะ").
"""

# Number pronunciation
_NUMBER_INSTRUCTIONS = """
Always speak numbers in Thai, never in English. For prices, quantities, and number of days, use natural Thai numeric words (e.g. 1290 → "หนึ่งพันสองร้อยเก้าสิบบาท", 14 days → "สิบสี่วัน"). For digit
sequences such as document number prefixes, codes, IDs, phone numbers, or years read aloud digit by digit, pronounce each digit in Thai
(e.g. "01" → "ศูนย์หนึ่ง", "2026" said as a year → "สองศูนย์สองหก", "0812345678" → "ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปด").
Never say "zero", "one", "two" in English.
"""

AGENT_INSTRUCTIONS = "\n\n".join(
    [
        _PERSONA,
        # _PRODUCT_INSTRUCTIONS,
        _DOCUMENT_INSTRUCTIONS,
        _WEB_SEARCH_INSTRUCTIONS,
        _NUMBER_INSTRUCTIONS,
    ]
)


class _TavusGreenScreen(tavus.AvatarSession):
    async def start(self, agent_session: AgentSession, room, **kwargs) -> None:
        _orig = self._api.create_conversation

        async def _with_greenscreen(**kw):
            props = dict(kw.pop("properties", None) or {})
            props["apply_greenscreen"] = True
            return await _orig(properties=props, **kw)

        self._api.create_conversation = _with_greenscreen
        await super().start(agent_session, room, **kwargs)


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

    @function_tool
    async def list_documents(self, context: RunContext) -> str:
        """List the uploaded documents (PDFs, text files) in the knowledge base.

        Use this when the user asks "what documents do we have?", "what files are
        uploaded?", or asks any question about uploaded files. Always call this
        first and present the names back to the user before calling read_document.
        """
        logger.info("Listing backend documents")
        docs = backend_docs.list_ready_documents()
        if not docs:
            return "No documents have been uploaded yet."
        return "\n".join(d.to_summary() for d in docs)

    @function_tool
    async def read_document(self, context: RunContext, document_id: str) -> str:
        """Read the text content of a specific uploaded document.

        Only call this AFTER list_documents and AFTER the user has chosen
        which document to read. The content is truncated for long files;
        the truncation note is included so you can tell the user.

        Args:
            document_id: The id from list_documents (e.g. "f1c69f7a-..."),
                or a partial-id prefix that uniquely identifies the document.
        """
        logger.info(f"Reading document: {document_id}")
        doc = backend_docs.get_document(document_id)
        if doc is None:
            for candidate in backend_docs.list_ready_documents():
                if candidate.id.startswith(document_id):
                    doc = candidate
                    break
        if doc is None:
            return f"Document with id '{document_id}' not found."
        try:
            text = backend_docs.read_document_text(doc)
        except FileNotFoundError:
            return f"The file for '{doc.name}' is missing on disk."
        except ValueError as exc:
            return str(exc)
        if not text:
            return f"'{doc.name}' contains no extractable text."
        return f"Content of {doc.name}:\n\n{text}"

    @function_tool
    async def web_search(self, context: RunContext, query: str) -> str:
        """Search the internet for general information about a topic, technology,
        or product type that is NOT in our shop catalog.

        Use this for questions like "what is USB-C?", "how does noise cancellation work?",
        or "what is the difference between SSD and HDD?". Do not use this for items
        in our catalog — use search_products for those.

        Args:
            query: English search query.
        """
        logger.info(f"Web searching: {query}")
        provider = web_search.get_default_provider()
        results = await provider.search(query, max_results=3)
        if not results:
            return f"No web results found for '{query}'."
        return "\n".join(r.to_summary() for r in results)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # products_db.init_db(PRODUCTS_DB_PATH)


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="deepgram/nova-2", language="th"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model=AGENT_MODEL),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
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

    # Add a virtual avatar to the session.
    # For other providers, see https://docs.livekit.io/agents/models/avatar/
    avatar = _TavusGreenScreen(
        replica_id=os.getenv("TAVUS_REPLICA_ID"),
        persona_id=os.getenv("TAVUS_PERSONA_ID"),
    )
    # To switch back to Beyond Presence, comment out the block above and use:
    # from livekit.plugins import bey
    # avatar = bey.AvatarSession(avatar_id=os.getenv("BEY_AVATAR_ID"))

    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

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
