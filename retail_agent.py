from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import json

# ── CONFIG ────────────────────────────────────────────
OPENAI_KEY = "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"

llm = ChatOpenAI(
    api_key=OPENAI_KEY,
    model="gpt-4o-mini",
    temperature=0.7
)

# ── STEP 1: ANALYSE CUSTOMER SEGMENT ─────────────────
analyse_prompt = ChatPromptTemplate.from_template("""
You are a retail data scientist at a major Australian supermarket.

Analyse this customer segment: {segment}

Provide:
1. KEY BUYING BEHAVIOURS: What do they typically buy? When? How often?
2. PAIN POINTS: What frustrations do they have?
3. MOTIVATIONS: What drives their purchase decisions?
4. LIFETIME VALUE: High/Medium/Low and why
5. BEST CHANNELS: Email, SMS, app notification, social media?

Be specific to Australian retail context. Keep it concise and data-driven.
""")

# ── STEP 2: GENERATE CAMPAIGN IDEAS ──────────────────
campaign_prompt = ChatPromptTemplate.from_template("""
You are a marketing AI strategist at a major Australian supermarket.

Based on this customer analysis:
{analysis}

Generate 3 targeted marketing campaign ideas for segment: {segment}

For each campaign provide:
- CAMPAIGN NAME
- CONCEPT (1 sentence)
- OFFER (specific discount or reward)
- TIMING (when to send)
- EXPECTED UPLIFT (% increase in spend)

Focus on personalisation and data-driven targeting.
""")

# ── STEP 3: WRITE EMAIL COPY ──────────────────────────
email_prompt = ChatPromptTemplate.from_template("""
You are an expert retail copywriter for a major Australian supermarket.

Customer segment: {segment}
Best campaign idea from: {campaigns}

Write a personalised marketing email for the BEST campaign idea.

Include:
- Subject line (compelling, under 50 chars)
- Preview text (under 100 chars)  
- Email body (friendly, personalised, Australian tone)
- Clear CTA (call to action)
- Personalisation tokens like [FIRST_NAME], [SUBURB]

Keep it short, punchy and conversion-focused.
""")

# ── STEP 4: SCORE & EVALUATE ──────────────────────────
score_prompt = ChatPromptTemplate.from_template("""
You are a marketing analytics AI at a major Australian supermarket.

Review this complete marketing campaign:

SEGMENT: {segment}
ANALYSIS: {analysis}
CAMPAIGNS: {campaigns}
EMAIL: {email}

Score this campaign out of 100 on:
1. PERSONALISATION SCORE (out of 25): How tailored is it?
2. RELEVANCE SCORE (out of 25): Does it match customer needs?
3. CONVERSION POTENTIAL (out of 25): Likelihood of purchase?
4. BRAND ALIGNMENT (out of 25): Fits supermarket brand values?

Also provide:
- TOTAL SCORE: X/100
- TOP STRENGTH: One thing done really well
- IMPROVEMENT: One specific thing to make it better
- RECOMMENDATION: Launch / Refine / Reject
""")

# ── AGENT PIPELINE ────────────────────────────────────
def run_marketing_agent(customer_segment: str):
    print(f"\n{'='*60}")
    print(f"🛒 RETAIL MARKETING INTELLIGENCE AGENT")
    print(f"{'='*60}")
    print(f"📊 Analysing segment: {customer_segment}")
    print(f"{'='*60}\n")

    # STEP 1 — Analyse
    print("🔍 STEP 1: Analysing customer segment...")
    analyse_chain = analyse_prompt | llm
    analysis = analyse_chain.invoke({"segment": customer_segment})
    analysis_text = analysis.content
    print(analysis_text)
    print("\n" + "-"*60 + "\n")

    # STEP 2 — Generate campaigns
    print("💡 STEP 2: Generating campaign ideas...")
    campaign_chain = campaign_prompt | llm
    campaigns = campaign_chain.invoke({
        "segment": customer_segment,
        "analysis": analysis_text
    })
    campaigns_text = campaigns.content
    print(campaigns_text)
    print("\n" + "-"*60 + "\n")

    # STEP 3 — Write email
    print("✉️  STEP 3: Writing personalised email copy...")
    email_chain = email_prompt | llm
    email = email_chain.invoke({
        "segment": customer_segment,
        "campaigns": campaigns_text
    })
    email_text = email.content
    print(email_text)
    print("\n" + "-"*60 + "\n")

    # STEP 4 — Score
    print("📈 STEP 4: Evaluating campaign effectiveness...")
    score_chain = score_prompt | llm
    score = score_chain.invoke({
        "segment": customer_segment,
        "analysis": analysis_text,
        "campaigns": campaigns_text,
        "email": email_text
    })
    score_text = score.content
    print(score_text)
    print("\n" + "="*60 + "\n")

    print("✅ AGENT COMPLETE — Full campaign generated!")
    print("="*60)

    return {
        "segment": customer_segment,
        "analysis": analysis_text,
        "campaigns": campaigns_text,
        "email": email_text,
        "score": score_text
    }

# ── RUN ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🛒 Welcome to the Retail Marketing Intelligence Agent!")
    print("Powered by LangChain + OpenAI GPT-4o-mini\n")

    # Example segments to try
    segments = [
        "young families with kids under 5 who buy organic and baby products weekly",
        "health-conscious millennials who buy protein and fitness products",
        "budget shoppers who buy home brand products and look for specials"
    ]

    print("Choose a customer segment:")
    for i, s in enumerate(segments, 1):
        print(f"{i}. {s}")

    print("\nOr type your own segment!")
    user_input = input("\nEnter segment number (1-3) or type your own: ").strip()

    if user_input in ['1', '2', '3']:
        segment = segments[int(user_input) - 1]
    else:
        segment = user_input

    result = run_marketing_agent(segment)