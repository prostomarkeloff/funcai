"""
Bot Farm Example — demonstrates funcai composition with .then() and .map()

A pipeline that reads fake posts from a pool and processes them through an agent.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel

from funcai import Dialogue, message, agent, tool
from funcai.std.providers.openai import OpenAIProvider
from combinators import batch
from kungfu import LazyCoroResult, Ok, Error, Result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Post:
    id: str
    author: str
    content: str
    likes: int


class PostAnalysis(BaseModel):
    """Agent's analysis of a post."""

    sentiment: str  # positive, negative, neutral
    topics: list[str]
    engagement_prediction: str  # low, medium, high
    suggested_response: str


@dataclass
class ProcessedPost:
    original: Post
    analysis: PostAnalysis
    response: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fake Post Pool (simulates external data source)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POST_POOL: list[Post] = [
    Post(
        id="p1",
        author="@techie_anna",
        content="Just discovered this amazing new Python library for AI! The API is so clean 🔥",
        likes=234,
    ),
    Post(
        id="p2",
        author="@grumpy_dev",
        content="Another day, another JS framework. When will this madness end?",
        likes=1892,
    ),
    Post(
        id="p3",
        author="@startup_guru",
        content="10 reasons why your startup will fail (thread) 🧵",
        likes=567,
    ),
    Post(
        id="p4",
        author="@coffee_coder",
        content="Is it just me or does every senior dev just Google everything too?",
        likes=4521,
    ),
    Post(
        id="p5",
        author="@rust_enjoyer",
        content="I rewrote my hello world in Rust. It's 10x faster now. You wouldn't understand.",
        likes=890,
    ),
]


def fetch_random_post() -> LazyCoroResult[Post, str]:
    """Simulate fetching a post from external source."""

    async def _fetch() -> Result[Post, str]:
        await asyncio.sleep(0.1)  # simulate network delay
        if random.random() < 0.1:  # 10% chance of failure
            return Error("Network error: could not fetch post")
        return Ok(random.choice(POST_POOL))

    return LazyCoroResult(_fetch)


def fetch_post_by_id(post_id: str) -> LazyCoroResult[Post, str]:
    """Fetch specific post by ID."""

    async def _fetch() -> Result[Post, str]:
        await asyncio.sleep(0.05)
        for post in POST_POOL:
            if post.id == post_id:
                return Ok(post)
        return Error(f"Post {post_id} not found")

    return LazyCoroResult(_fetch)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

provider = OpenAIProvider(model="gpt-4o-mini")

# Bot's memory of previously seen posts
seen_posts: set[str] = set()


@tool("Check if we've already processed this post")
def check_if_seen(post_id: str) -> bool:
    return post_id in seen_posts


@tool("Mark a post as processed")
def mark_as_seen(post_id: str) -> str:
    seen_posts.add(post_id)
    return f"Marked {post_id} as seen"


@tool("Get engagement tier based on likes")
def get_engagement_tier(likes: int) -> str:
    if likes < 100:
        return "low"
    elif likes < 1000:
        return "medium"
    else:
        return "viral"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_post(post: Post) -> LazyCoroResult[PostAnalysis, str | Any]:
    """Analyze a post using the agent."""

    dialogue = Dialogue(
        [
            message.system(
                text="""You are a social media analyst bot.
                Analyze posts and provide structured insights.
                Use tools to check post history and engagement metrics."""
            ),
            message.user(
                text=f"""Analyze this post:

Author: {post.author}
Content: {post.content}
Likes: {post.likes}
Post ID: {post.id}

First check if we've seen this post, get its engagement tier, and then provide analysis."""
            ),
        ]
    )

    # Agent returns AgentResponse, we need to extract and parse
    return agent(
        dialogue,
        provider,
        tools=[check_if_seen, mark_as_seen, get_engagement_tier],
        schema=PostAnalysis,
    ).map(lambda r: r.parsed)


def format_result(processed: ProcessedPost) -> str:
    """Format processed post for display."""
    return f"""
┌─────────────────────────────────────────────────────────────
│ 📝 Post by {processed.original.author}
│ "{processed.original.content}"
│ ❤️  {processed.original.likes} likes
├─────────────────────────────────────────────────────────────
│ 📊 Analysis:
│    Sentiment: {processed.analysis.sentiment}
│    Topics: {', '.join(processed.analysis.topics)}
│    Engagement: {processed.analysis.engagement_prediction}
├─────────────────────────────────────────────────────────────
│ 💬 Bot Response:
│    {processed.response}
└─────────────────────────────────────────────────────────────"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Pipeline — The Magic ✨
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def process_single_post() -> LazyCoroResult[str, str]:
    """
    Full pipeline using .then() and .map() composition:

    fetch_random_post()
        → analyze with agent
        → create ProcessedPost
        → format for display
    """
    return (
        fetch_random_post()
        .then(
            lambda post: analyze_post(post).map(
                lambda analysis: ProcessedPost(
                    original=post,
                    analysis=analysis,
                    response=analysis.suggested_response,
                )
            )
        )
        .map(format_result)
        .map_err(lambda e: f"Pipeline failed: {e}")
    )


def process_post_by_id(post_id: str) -> LazyCoroResult[str, str]:
    """Process a specific post by ID."""
    return (
        fetch_post_by_id(post_id)
        .then(
            lambda post: analyze_post(post).map(
                lambda analysis: ProcessedPost(
                    original=post,
                    analysis=analysis,
                    response=analysis.suggested_response,
                )
            )
        )
        .map(format_result)
    )


async def run_bot_farm(n_posts: int = 3) -> None:
    """Run the bot farm on multiple posts."""
    print("🤖 Bot Farm starting...\n")

    # Process multiple posts using batch combinator
    results = await batch(
        list(range(n_posts)),
        handler=lambda _: process_single_post(),
        concurrency=2,
    )

    match results:
        case Ok(formatted_posts):
            for formatted in formatted_posts:
                print(formatted)
                print()
        case Error(e):
            print(f"❌ Batch processing failed: {e}")


async def interactive_mode() -> None:
    """Interactive mode — process specific posts or random ones."""
    print("🤖 Bot Farm Interactive Mode")
    print("Commands: 'random', 'p1'-'p5' (post IDs), 'quit'\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if cmd == "quit":
            break
        elif cmd == "random":
            result = await process_single_post()
            match result:
                case Ok(formatted):
                    print(formatted)
                case Error(e):
                    print(f"❌ {e}")
        elif cmd.startswith("p"):
            result = await process_post_by_id(cmd)
            match result:
                case Ok(formatted):
                    print(formatted)
                case Error(e):
                    print(f"❌ {e}")
        else:
            print("Unknown command. Try 'random', 'p1'-'p5', or 'quit'")


async def main() -> None:
    """Entry point — runs batch mode then interactive."""
    await run_bot_farm(n_posts=2)
    await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())






