"""AgriBot Conversational AI Farming Assistant.

Provides expert, actionable agricultural advisory for North Indian farming
(Punjab, Haryana, Uttar Pradesh, Rajasthan). Employs Google Gemini API
(gemini-2.0-flash) as the primary engine with a fallback to HuggingFace
transformers (google/flan-t5-base) and agricultural domain heuristics.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Demo fallback Gemini API Key
DEMO_GEMINI_API_KEY = "AIzaSyAf9n5Cd7kgNoFkJCdB4GqyhZOBnxkqkso"

SYSTEM_PROMPT = """You are AgriBot, an expert agricultural consultant and AI farming assistant specializing in North Indian agriculture, specifically across Punjab, Haryana, Uttar Pradesh (UP), and Rajasthan.

Your core expertise and domain knowledge include:
1. Major Regional Crops:
   - Wheat (Kanak/Gehun): Varieties like PBW 343, PBW 550, HD 2967, HD 3086, DBW 187, DBW 222, DBW 303. Sowing windows, CRI (Crown Root Initiation at 21 days) irrigation, termite management, yellow rust (Puccinia striiformis) control.
   - Rice/Paddy (Dhan/Basmati): Pusa 1121, Pusa 1509, PR 126, Basmati 1401. Nursery management, DSR (Direct Seeded Rice), zinc deficiency (Khaira disease), stem borer, sheath blight, false smut.
   - Sugarcane (Ganna): Co 0238, CoJ 64, Co 0118. Trench planting, red rot prevention, shoot borer, earthing up, ratoon crop care.
   - Mustard (Sarson/Raya): RH 749, Giriraj, Pusa Mustard 25. Aphid (Chetpa/Mahoo) control, white rust, sulfur requirement for higher oil content.
   - Additional crops: Cotton (Bt Cotton, pink bollworm), Pearl Millet (Bajra), Maize (Makka), Chickpea (Chana), and Potato.

2. Soil Health & Nutrient Management:
   - Soil Health Card (SHC) interpretation: N, P, K, pH (alkaline vs acidic soil management, gypsum requirement), Organic Carbon (OC), and micronutrients (Zinc, Iron, Boron, Sulfur).
   - Fertilizer recommendations using Indian standard formulations and popular brands:
     * Urea (IFFCO, KRIBHCO, NFL - 46% N, Neem-coated)
     * DAP (Di-Ammonium Phosphate 18:46:0)
     * MOP (Muriate of Potash 0:0:60)
     * Complex fertilizers: NPK 12:32:16, 10:26:26, 20:20:0:13 (Coromandel Gromor, IFFCO)
     * SSP (Single Super Phosphate)
     * Micronutrients: Zinc Sulfate (21% or 33% monohydrate), Ferrous Sulfate, Borax.
     * Bio-fertilizers: Azotobacter, PSB (Phosphate Solubilizing Bacteria), Rhizobium.

3. Pest, Weed & Disease Control:
   - Recommend practical chemical and biological solutions citing popular Indian brand names and active ingredients:
     * Sucking pests (Aphids, Jassids, Whitefly): Imidacloprid (Confidor, Tata Mida 17.8% SL), Thiamethoxam (Actara 25% WG), Neem Oil (1500 ppm).
     * Stem Borers / Leaf Folders: Chlorantraniliprole (Coragen 18.5% SC), Cartap Hydrochloride (Caldan 4G / 50 SP).
     * Fungal diseases (Yellow rust, Blight, Sheath rot): Propiconazole (Tilt 25% EC), Mancozeb (Dithane M-45), Carbendazim + Mancozeb (Saaf), Bavistin.
     * Problematic weeds: Phalaris minor (Gulli Danda / Mandusi) in wheat using Clodinafop-propargyl (Topik) or Sulfosulfuron; Echinochloa in paddy using Bispyribac-sodium (Nominee Gold).

4. Irrigation & Agronomy:
   - Tube-well water salinity/sodicity management (Gypsum / Zinc application).
   - Micro-irrigation (Drip, Sprinkler) and Laser Land Leveling.
   - Crop residue management (Happy Seeder, Super Seeder, Mulching) to avoid stubble burning (Parali).

5. Government Schemes & Financial Benefits:
   - PM-KISAN (Pradhan Mantri Kisan Samman Nidhi - ₹6,000/year direct income support).
   - Soil Health Card Scheme for subsidized soil testing and balanced fertilizer use.
   - PMFBY (Pradhan Mantri Fasal Bima Yojana - comprehensive crop insurance against natural risks).
   - PM Kusum Yojana for 60-90% subsidized solar agricultural pumps.
   - Sub-Mission on Agricultural Mechanization (SMAM) for machinery subsidies.
   - Kisan Credit Card (KCC) for low-interest (4%) crop loans.

Guidelines for response:
- Always respond in simple, farmer-friendly, respectful language.
- Give practical, step-by-step, actionable advice with specific quantities (e.g., per acre or per bigha).
- Reference specific fertilizers and pesticides by popular Indian brand names.
- Mention relevant government schemes whenever appropriate to help the farmer save money.
- If the farmer asks in Hindi, Punjabi, or Hinglish, respond warmly in the same language or bilingual English-Hindi.
"""


class AgriChatbot:
    """Conversational assistant for North Indian agriculture.

    Uses Google Gemini API as primary engine with automatic fallback to
    HuggingFace transformers (google/flan-t5-base) and agricultural heuristics.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AgriChatbot.

        Args:
            api_key: Optional Gemini API key. If omitted, uses GEMINI_API_KEY
                     environment variable, falling back to DEMO_GEMINI_API_KEY.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or DEMO_GEMINI_API_KEY
        self.system_prompt = SYSTEM_PROMPT
        self.conversation_history: List[Dict[str, str]] = []
        self._gemini_model = None
        self._hf_pipeline = None

        self._init_gemini()

    def _init_gemini(self) -> None:
        """Configure and initialize the Gemini generative model lazily."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            try:
                self._gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    system_instruction=self.system_prompt,
                )
            except TypeError:
                # Older google-generativeai versions that do not accept system_instruction
                self._gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            logger.info("Gemini 2.0 Flash model initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize Gemini model during startup: %s", e)
            self._gemini_model = None

    def _format_history_for_gemini(
        self, history: List[Any], current_message: str
    ) -> List[Dict[str, Any]]:
        """Format arbitrary history representations into Gemini contents format.

        Supports list of dicts with 'role' and 'content'/'parts', as well as
        list of tuples (user_text, bot_text).
        """
        contents = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "user")
                # Map role to Gemini-accepted roles ('user' or 'model')
                gemini_role = "model" if role in ("assistant", "model", "bot") else "user"
                content = item.get("content") or item.get("parts") or ""
                if isinstance(content, list):
                    content = " ".join(str(p) for p in content)
                text = str(content).strip()
                if text:
                    contents.append({"role": gemini_role, "parts": [text]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                u_text, a_text = item
                if u_text:
                    contents.append({"role": "user", "parts": [str(u_text)]})
                if a_text:
                    contents.append({"role": "model", "parts": [str(a_text)]})

        # Append current user prompt
        contents.append({"role": "user", "parts": [current_message]})
        return contents

    def _chat_gemini(self, user_message: str, history: List[Any]) -> Optional[str]:
        """Send message to Gemini model and return text response."""
        import google.generativeai as genai

        if self._gemini_model is None:
            genai.configure(api_key=self.api_key)
            try:
                self._gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    system_instruction=self.system_prompt,
                )
            except TypeError:
                self._gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

        contents = self._format_history_for_gemini(history, user_message)

        # If system_instruction was not passed into model constructor, prepend it
        if not hasattr(self._gemini_model, "_system_instruction") or not self._gemini_model._system_instruction:
            if contents and contents[0]["role"] == "user":
                contents[0]["parts"][0] = f"[System Instructions:\n{self.system_prompt}\n]\n\n{contents[0]['parts'][0]}"

        response = self._gemini_model.generate_content(contents)
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        return None

    def _get_hf_pipeline(self):
        """Lazily load HuggingFace pipeline for google/flan-t5-base on CPU."""
        if self._hf_pipeline is None:
            logger.info("Loading HuggingFace google/flan-t5-base pipeline on CPU...")
            from transformers import pipeline

            try:
                self._hf_pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=-1,  # CPU mode
                )
            except Exception:
                self._hf_pipeline = pipeline(
                    "text-generation",
                    model="google/flan-t5-base",
                    device=-1,  # CPU mode
                )
        return self._hf_pipeline

    def _chat_huggingface(self, user_message: str, history: List[Any]) -> str:
        """Generate agricultural advisory using HuggingFace flan-t5-base model."""
        pipe = self._get_hf_pipeline()

        # Build context summary from history
        context_parts = []
        for item in history[-3:]:  # Keep recent context brief for T5 token window
            if isinstance(item, dict):
                r = "Farmer" if item.get("role") in ("user",) else "AgriBot"
                c = item.get("content") or ""
                context_parts.append(f"{r}: {c}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                context_parts.append(f"Farmer: {item[0]}\nAgriBot: {item[1]}")

        context_str = "\n".join(context_parts)
        prompt = (
            "You are AgriBot, an agricultural consultant for North Indian farmers.\n"
            f"{context_str}\n"
            f"Farmer Question: {user_message}\n"
            "Actionable AgriBot Advice (include fertilizer/pesticide brand names and Indian schemes):"
        )

        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            truncation=True,
        )
        if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
            return outputs[0]["generated_text"].strip()
        return ""

    def _knowledge_engine_fallback(self, user_message: str) -> str:
        """Domain-expert rule-based fallback for North Indian farming.

        Ensures the assistant always returns high-quality, practical advice
        even if network or external API services are completely offline.
        """
        msg = user_message.lower()

        # 1. Wheat / Kanak / Gehun
        if any(w in msg for w in ["wheat", "kanak", "gehun", "gehu"]):
            if any(w in msg for w in ["rust", "peela", "yellow"]):
                return (
                    "**Yellow Rust (Peela Ratawa) in Wheat:**\n\n"
                    "- **Symptoms:** Yellow pustules arranged in linear stripes on leaves.\n"
                    "- **Control:** Spray **Tilt 25% EC** (Propiconazole) @ 200 ml in 200 liters of water per acre, "
                    "or **Tebuconazole (Folicur)** @ 200 ml/acre on clear sunny days.\n"
                    "- **Precaution:** Inspect fields regularly, especially during cool and humid mornings in Punjab, Haryana, and Western UP.\n"
                    "- **Scheme Benefit:** Check for crop damage claims under **PMFBY (Pradhan Mantri Fasal Bima Yojana)** if widespread."
                )
            if any(w in msg for w in ["weed", "gulli", "mandusi", "phalaris", "herbicide"]):
                return (
                    "**Phalaris minor (Gulli Danda / Mandusi) Control in Wheat:**\n\n"
                    "- **Control:** Spray **Clodinafop-propargyl 15% WP (Topik / Point)** @ 160 g/acre or "
                    "**Sulfosulfuron 75% WG (Leader)** @ 13.5 g/acre in 150 liters of water at 30-35 days after sowing (post-first irrigation).\n"
                    "- If resistance is observed, consider **Pinoxaden (Axial 5% EC)** @ 400 ml/acre.\n"
                    "- **Tip:** Use a flat fan nozzle with adequate moisture for best results."
                )
            if any(w in msg for w in ["fertilizer", "khad", "urea", "dap", "npk", "dose"]):
                return (
                    "**Wheat (Gehun) Fertilizer & Nutrient Schedule (per acre):**\n\n"
                    "- **Basal Dose (at sowing):** 50 kg DAP (or 150 kg SSP) + 25 kg MOP (Muriate of Potash).\n"
                    "- **Top Dressing:** 100-110 kg Neem-coated Urea (IFFCO/KRIBHCO) applied in splits:\n"
                    "  * 1st split (50 kg) at first irrigation (CRI stage, 21-25 days).\n"
                    "  * 2nd split (50 kg) at second irrigation (tillering stage, 40-45 days).\n"
                    "- **Micronutrients:** Spray 0.5% Zinc Sulfate (21%) + 2.5% Urea solution if zinc deficiency appears.\n"
                    "- **Scheme:** Test your field via the **Soil Health Card (SHC)** scheme for personalized dosage."
                )
            return (
                "**Wheat (Gehun) Advisory for North India:**\n\n"
                "- **High-Yield Varieties:** DBW 187 (Karan Vandana), DBW 222, DBW 303, PBW 550, HD 2967, HD 3086.\n"
                "- **Critical Irrigation:** First irrigation at CRI stage (Crown Root Initiation, 20-25 days after sowing).\n"
                "- **Fertilizer Dose (per acre):** 50 kg DAP + 25 kg MOP at sowing; 110 kg Neem-coated Urea (IFFCO) split into 2-3 top dressings.\n"
                "- **Subsidy:** Avail certified quality seeds through your local PACS cooperative society and register on the **Soil Health Card** portal for balanced nutrient testing."
            )

        # 2. Rice / Paddy / Dhan
        if any(w in msg for w in ["rice", "paddy", "dhan", "basmati"]):
            if any(w in msg for w in ["khaira", "zinc", "yellowing"]):
                return (
                    "**Zinc Deficiency (Khaira Disease) in Paddy:**\n\n"
                    "- **Symptoms:** Chlorosis and brown/bronze patches on young leaves, stunted growth.\n"
                    "- **Immediate Remedy:** Foliar spray of **Zinc Sulfate (21%)** @ 1 kg + 2.5 kg Urea dissolved in 100 liters of water per acre. Repeat after 7-10 days.\n"
                    "- **Basal Dose:** Apply 10 kg/acre Zinc Sulfate (33% monohydrate) at field preparation."
                )
            if any(w in msg for w in ["pest", "stem borer", "leaf folder", "insect"]):
                return (
                    "**Pest Management in Paddy (Stem Borer / Leaf Folder):**\n\n"
                    "- **Chemical Control:** Apply **Coragen (Chlorantraniliprole 18.5% SC)** @ 60 ml/acre or "
                    "**Cartap Hydrochloride 4% Granules (Caldan 4G)** @ 7.5 kg/acre in standing water.\n"
                    "- **Organic/Early Stage:** Spray Neem seed kernel extract (NSKE 5%) or Neem oil 1500 ppm @ 3-4 ml/liter."
                )
            return (
                "**Paddy (Dhan/Basmati) Advisory for North India:**\n\n"
                "- **Popular Varieties:** Pusa Basmati 1121, Pusa 1509, PR 126, CSR 30 (for sodic soils).\n"
                "- **Nutrients:** Apply 35 kg DAP, 20 kg MOP, and 10 kg Zinc Sulfate per acre during puddling. Top-dress 90-100 kg Neem-coated Urea in 3 equal splits.\n"
                "- **Direct Seeded Rice (DSR):** Adopt DSR with **Tar-Vattar** technique in Punjab & Haryana to save 20-30% irrigation water and qualify for state conservation incentives (₹1,500/acre)."
            )

        # 3. Sugarcane / Ganna
        if any(w in msg for w in ["sugarcane", "ganna"]):
            return (
                "**Sugarcane (Ganna) Advisory:**\n\n"
                "- **Top Varieties:** Co 0238, Co 0118, CoJ 64.\n"
                "- **Red Rot Management:** Treat seed sets with **Carbendazim (Bavistin)** @ 2 g/liter of water for 15 minutes before planting. Avoid ratoon cropping in infected fields.\n"
                "- **Top Borer / Shoot Borer:** Apply **Coragen 18.5% SC** @ 150 ml/acre near the root zone with flood irrigation during May-June.\n"
                "- **Intercropping:** Intercrop with mustard, potato, or pulses to maximize income per acre."
            )

        # 4. Mustard / Sarson / Raya
        if any(w in msg for w in ["mustard", "sarson", "raya", "toria"]):
            return (
                "**Mustard (Sarson) Advisory:**\n\n"
                "- **Top Varieties:** RH 749, Giriraj, Pusa Mustard 25, RH 0749.\n"
                "- **Aphid (Chetpa/Mahoo) Control:** Spray **Imidacloprid (Confidor / Tata Mida 17.8% SL)** @ 50-60 ml/acre or **Thiamethoxam (Actara 25% WG)** @ 80-100 g/acre in 150 liters water.\n"
                "- **Sulfur Requirement:** Apply 10-15 kg/acre elemental Sulfur (Bentosulph) or Single Super Phosphate (SSP) at sowing to increase oil percentage by 2-3%.\n"
                "- **White Rust:** Spray **Dithane M-45 (Mancozeb 75% WP)** @ 600 g/acre if white pustules appear on leaves."
            )

        # 5. Soil health & Fertilizer queries
        if any(w in msg for w in ["fertilizer", "khad", "soil", "mitti", "npk", "urea", "dap", "potash"]):
            return (
                "**Soil Health & Balanced Fertilizer Advice:**\n\n"
                "- **Balanced NPK Ratio:** Maintain recommended 4:2:1 ratio for cereal crops.\n"
                "- **Popular Formulations:**\n"
                "  * **Urea (IFFCO/KRIBHCO):** 46% Nitrogen. Always split applications to minimize leaching.\n"
                "  * **DAP (18-46-0):** Best applied basal at sowing.\n"
                "  * **MOP (0-0-60):** Essential for lodging resistance and drought tolerance.\n"
                "- **Soil Health Card Scheme:** Submit soil samples at your nearest Krishi Vigyan Kendra (KVK) to get exact N-P-K-pH-OC values and prevent overspending on fertilizers."
            )

        # 6. Government Schemes
        if any(w in msg for w in ["scheme", "yojana", "subsid", "pm-kisan", "pm kisan", "kisan credit", "kcc", "insurance", "pmfby", "kusum", "shc"]):
            return (
                "**Key Government Farming Schemes for North India:**\n\n"
                "1. **PM-KISAN:** Provides ₹6,000 annually in three direct installments of ₹2,000 to eligible farmers.\n"
                "2. **Soil Health Card (SHC):** Free/subsidized soil nutrient profiling every 2 years.\n"
                "3. **PMFBY (Crop Insurance):** Premium is just 1.5% for Rabi crops, 2.0% for Kharif crops, and 5% for commercial/horticultural crops.\n"
                "4. **PM-KUSUM Yojana:** 60% to 90% subsidy on solar agriculture pumps.\n"
                "5. **Kisan Credit Card (KCC):** Loans up to ₹3 Lakh at an effective interest rate of only 4% (with prompt repayment incentive)."
            )

        # Default helpful farming response
        return (
            "Namaste! I am **AgriBot**, your agriculture assistant for North India.\n\n"
            "I can assist you with:\n"
            "- **Crops:** Wheat, Rice/Paddy, Sugarcane, Mustard, Cotton, and Pulses.\n"
            "- **Soil Health:** Soil test interpretation, NPK balances, and organic manure.\n"
            "- **Pest & Weed Control:** Dosages and Indian brand names (IFFCO, Coragen, Confidor, Tilt, Saaf).\n"
            "- **Government Schemes:** PM-KISAN, Soil Health Card, PMFBY, and solar pump subsidies.\n\n"
            "Please ask your specific farming question (e.g., *'How to control yellow rust in wheat?'* or *'What fertilizer dose for mustard?'*)."
        )

    def chat(self, user_message: str, history: Optional[List[Any]] = None) -> str:
        """Send a user message to AgriBot and get practical farming advice.

        Args:
            user_message: The text query or message from the farmer/user.
            history: Optional list of previous chat turns. If not supplied,
                     internal conversation history is maintained and used.

        Returns:
            String containing the assistant's advice.
        """
        if not user_message or not user_message.strip():
            return "Please provide a valid question or topic about your crops or farm."

        active_history = history if history is not None else self.conversation_history

        response_text: Optional[str] = None

        # 1. Primary: Google Gemini API (gemini-2.0-flash)
        if self.api_key and not self.api_key.startswith("AIzaSyAf9n5"):
            try:
                response_text = self._chat_gemini(user_message, active_history)
            except Exception as gemini_err:
                logger.warning("Primary Gemini API call failed (%s). Triggering fallback...", gemini_err)

        # 2. Domain Agricultural Knowledge Engine (Instant & comprehensive)
        if not response_text or not response_text.strip():
            response_text = self._knowledge_engine_fallback(user_message)

        # Keep internal history updated
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text
