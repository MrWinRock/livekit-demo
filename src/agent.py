import logging
import os
from pathlib import Path

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
    room_io,
)
from livekit.plugins import ai_coustics, bey, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import products_db
import web_search

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_MODEL = "openai/gpt-5.3-chat-latest"
PRODUCTS_DB_PATH = Path(__file__).parent.parent / "data" / "products.db"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI shop assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            Always respond in Thai language.
            You are female. Always use female polite particles: end sentences with "ค่ะ" or "นะคะ", use "ดิฉัน" or "หนู" to refer to yourself, and use "คุณ" when addressing the user.
            You help customers find products in our shop. When the user asks about products, prices, or what is available, use the search_products or list_all_products tools to get accurate information from the catalog. Never invent products or prices.
            For general questions about a technology, what a product type does, how something works, or any topic outside our shop catalog, use the web_search tool. Do not use web_search for items already in our catalog — use search_products for those.
            Web search takes a couple of seconds, so when you decide to use it, briefly tell the user you are looking it up (e.g. "ขอดูข้อมูลก่อนนะคะ").
            Speak prices naturally in Thai baht (e.g. "หนึ่งพันสองร้อยเก้าสิบบาท" for 1290 baht). Translate English product names to Thai when speaking, but pass English keywords to the search tool.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.
            When greeting the user, ask them what they would like help with today.""",
        )

    @function_tool
    async def search_products(self, context: RunContext, query: str) -> str:
        """Search the product catalog by name or description keyword.

        Use this when the user asks about a specific product, type of product,
        or feature. Pass English keywords (e.g. "headphones", "bluetooth", "cable").

        Args:
            query: English keyword to search for in product name or description.
        """
        logger.info(f"Searching products for: {query}")
        results = products_db.search_products(PRODUCTS_DB_PATH, query)
        if not results:
            return f"No products found matching '{query}'."
        return "\n".join(p.to_summary() for p in results)

    @function_tool
    async def list_all_products(self, context: RunContext) -> str:
        """List every product in the catalog.

        Use this when the user asks what is available, what you sell, or wants
        to browse the full catalog.
        """
        logger.info("Listing all products")
        products = products_db.list_all_products(PRODUCTS_DB_PATH)
        return "\n".join(p.to_summary() for p in products)

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
    products_db.init_db(PRODUCTS_DB_PATH)


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

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    avatar = bey.AvatarSession(
        avatar_id=os.getenv("BEY_AVATAR_ID"),
    )

    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user warmly in Thai and ask what you can help them with today."
    )


if __name__ == "__main__":
    cli.run_app(server)
