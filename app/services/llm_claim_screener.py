import json
from typing import Dict, List, Tuple
import os
from app.models.claim import Claim
from app.services.openai_service import OpenAIService
import asyncio
from textwrap import dedent
import logging
from asyncio import Semaphore
from asyncio import TimeoutError
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClaimScreener:
    PLAUSIBILITY_LEVELS = [
        "impossible",
        "highly implausible",
        "somewhat implausible", 
        "suspect",
        "potentially plausible",
        "plausible",
        "highly plausible", 
        "supported by evidence",
        "highly supported",
        "factual"
    ]

    def __init__(self, loop=None):
        self.openai_service = OpenAIService(loop=loop)
        self.semaphore = Semaphore(100)

    def update_claim_status(self, batch_id: str, claim_id: str, status: str, additional_info: dict = None):
        # Create directory if it doesn't exist
        directory = os.path.join('saved_jobs', batch_id)
        os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, f"{claim_id}.txt")
        
        # Create initial content if file doesn't exist
        if not os.path.exists(file_path):
            initial_content = {
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
            if additional_info is not None:
                initial_content['additional_info'] = additional_info
            
            with open(file_path, 'w') as f:
                json.dump(initial_content, f, indent=2)
            return
        
        # Update existing file
        with open(file_path, 'r+') as f:
            content = json.load(f)
            content['status'] = status
            if additional_info is not None:
                content['additional_info'] = additional_info
            f.seek(0)
            json.dump(content, f, indent=2)
            f.truncate()

    async def screen_claim(self, claim: Claim, batch_id: str, claim_id: str, loop=None) -> Dict:
        """Perform comprehensive LLM-based screening of the claim"""
        if loop is None:
            loop = asyncio.get_event_loop()
            
        # If we were initialized without a loop, but got one in screen_claim,
        # create a new OpenAIService with the correct loop
        if self.openai_service._loop is None and loop is not None:
            self.openai_service = OpenAIService(loop=loop)
            
        try:
            logger.info(f"Starting mechanism generation for claim {claim_id}")
            
            self.update_claim_status(batch_id, claim_id, "generating_supporting_mechanisms")
            supporting_task = self._generate_mechanisms(claim.text, mechanism_type="supporting")

            self.update_claim_status(batch_id, claim_id, "generating_contradicting_mechanisms")
            contradicting_task = self._generate_mechanisms(claim.text, mechanism_type="contradicting")
            
            supporting_mechanisms, contradicting_mechanisms = await asyncio.gather(
                supporting_task, contradicting_task
            )
            
            logger.info(f"Generated mechanisms for claim {claim_id}")

            self.update_claim_status(batch_id, claim_id, "analyzing_theoretical_basis")
            theoretical_analysis = await self._analyze_theoretical_basis(
                claim.text, supporting_mechanisms, contradicting_mechanisms)

            self.update_claim_status(batch_id, claim_id, "synthesizing_findings")
            synthesis = await self._synthesize_findings(
                claim.text,
                supporting_mechanisms,
                contradicting_mechanisms,
                theoretical_analysis
            )

            # Create the complete report
            report = {
                "claim": claim.text,
                "overall_rating": synthesis["rating"],
                "plausibility_level": synthesis["plausibility_level"],
                "summary": synthesis["summary"],
                "supporting_mechanisms": supporting_mechanisms,
                "contradicting_mechanisms": contradicting_mechanisms,
                "theoretical_analysis": theoretical_analysis,
                "key_uncertainties": synthesis["key_uncertainties"],
                "suggested_searches": synthesis["suggested_searches"],
                "usage_stats": self.openai_service.get_usage_stats()
            }

            self.update_claim_status(batch_id, claim_id, "processed", additional_info=report)
            logger.info(f"Saving report for claim {claim_id}")
            self._save_report(batch_id, claim_id, report)
            logger.info(f"Completed all processing for claim {claim_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error in screen_claim for {claim_id}: {str(e)}")
            self.update_claim_status(batch_id, claim_id, "error", additional_info={"error": str(e)})
            self._save_report(batch_id, claim_id, {
                "claim": claim.text,
                "error": str(e),
                "status": "error"
            })
            raise

    async def _generate_mechanisms(self, claim_text: str, mechanism_type: str) -> List[Dict]:
        """Generate and evaluate potential mechanisms that could support or contradict the claim"""
        try:
            logger.info(f"Generating {mechanism_type} mechanisms")
            direction = "support" if mechanism_type == "supporting" else "contradict"
            
            system_prompt = dedent(f"""
                You are an expert scientist tasked with identifying potential mechanisms that could {direction} a claim.
                For each mechanism:
                1. Consider biological, chemical, physical, and other relevant scientific principles
                2. Think creatively but remain grounded in scientific possibility
                3. Consider both direct and indirect pathways
                4. Include both obvious and non-obvious mechanisms
                5. Consider different levels of analysis (molecular, cellular, systemic, etc.)
                
                Respond with a JSON array of 3 to 5 mechanisms, where each mechanism has:
                {{
                    "mechanism_subclaim": "A one sentence statement of the mechanism",
                    "explanation": "Detailed scientific explanation",
                    "type": "{mechanism_type}",
                    "causal_chain": ["Detailed steps in the mechanism"],
                    "key_assumptions": ["Critical assumptions that must hold"],
                    "plausibility_assessment": {{
                        "level": "One of: impossible, highly implausible, somewhat implausible, suspect, potentially plausible, plausible, highly plausible, supported by evidence, highly supported, factual",
                        "reasoning": "Detailed explanation of the plausibility rating"
                    }}
                }}
            """).strip()

            prompt = dedent(f"""
                Identify and evaluate potential mechanisms that could {direction} the following claim:
                
                {claim_text}
                
                Generate 3 to 5 distinct mechanisms, ranging from obvious to non-obvious but scientifically possible.
                Provide a detailed evaluation of each mechanism's plausibility and implications.
            """).strip()

            try:
                response = await self.openai_service.generate_json_async(
                    prompt, 
                    system_prompt,
                    model="gpt-4o"
                )
                logger.info(f"Received response for {mechanism_type} mechanisms")
                
                # Add logging for token usage from the last API call
                usage_stats = self.openai_service.get_usage_stats()
                logger.info(f"Mechanism generation token usage - Last call total tokens: "
                           f"{usage_stats['total_tokens']}, "
                           f"Running total cost: ${usage_stats['total_cost']:.4f}")
                
                # Ensure we get a list of mechanisms
                if isinstance(response, list):
                    return response
                elif isinstance(response, dict) and 'mechanisms' in response:
                    return response['mechanisms']
                else:
                    # Return a single mechanism in a list if we got a single dict
                    if isinstance(response, dict):
                        return [response]
                    # Return empty list as fallback
                    return []
            except TimeoutError:
                logger.warning(f"Timeout generating {mechanism_type} mechanisms after 3 retries")
                return []
                
        except Exception as e:
            logger.error(f"Error generating {mechanism_type} mechanisms: {str(e)}")
            return []

    async def _analyze_theoretical_basis(
        self, 
        claim_text: str, 
        supporting_mechanisms: List[Dict],
        contradicting_mechanisms: List[Dict]
    ) -> Dict:
        """Analyze the theoretical foundation of the claim"""

        plausibility_levels_string = ", ".join(self.PLAUSIBILITY_LEVELS)

        system_prompt = dedent("""
            You are a theoretical scientist analyzing the scientific foundations of a claim.
            Consider:
            1. Theoretical frameworks that could explain the claim
            2. Known scientific principles that apply
            3. Potential violations of established theories
            4. Mathematical or logical constraints
            5. Systems-level implications
            
            Respond with a JSON object containing:
            {
                "theoretical_frameworks": ["Relevant theoretical frameworks"],
                "key_principles": ["Scientific principles that apply"],
                "theoretical_constraints": ["Theoretical limitations or constraints"],
                "systems_analysis": "Analysis of systems-level implications",
                "mathematical_constraints": "Any mathematical or logical constraints",
                "theoretical_plausibility": {
                    "level": f"One of: {plausibility_levels_string}",
                    "reasoning": "Explanation of theoretical plausibility"
                }
            }
        """).strip()

        prompt = dedent(f"""
            Analyze the theoretical basis for the following claim:
            
            Claim: {claim_text}
            
            Supporting Mechanisms Summary:
            {json.dumps([{'mechanism_subclaim': m['mechanism_subclaim'], 'explanation': m['explanation'], 'plausibility_assessment': m['plausibility_assessment']} 
                        for m in supporting_mechanisms], indent=2)}
            
            Contradicting Mechanisms Summary:
            {json.dumps([{'mechanism_subclaim': m['mechanism_subclaim'], 'explanation': m['explanation'], 'plausibility_assessment': m['plausibility_assessment']} 
                        for m in contradicting_mechanisms], indent=2)}
            
            Provide a theoretical analysis of this claim's scientific foundation.
        """).strip()

        return await self.openai_service.generate_json_async(prompt, system_prompt)

    async def _synthesize_findings(
        self,
        claim_text: str,
        supporting_mechanisms: List[Dict],
        contradicting_mechanisms: List[Dict],
        theoretical_analysis: Dict
    ) -> Dict:
        """Synthesize all findings into a final assessment"""
        system_prompt = dedent("""
            You are a senior scientist synthesizing multiple analyses into a final assessment.
            Consider:
            1. Strength and plausibility of supporting mechanisms
            2. Strength and plausibility of contradicting mechanisms
            3. Theoretical foundations
            4. Overall coherence of the evidence
            5. Key uncertainties and gaps
            
            For suggested searches, create precise search queries that would find the most relevant scientific papers
            on OpenAlex or Semantic Scholar. Focus on specific mechanisms, methodologies, or theoretical aspects
            that would help validate or refute the claim.
            
            Provide a JSON object containing:
            {
                "rating": "Integer from -10 to 10",
                "plausibility_level": f"One of: {plausibility_levels_string}",
                "summary": "Detailed summary of the assessment",
                "key_uncertainties": ["Critical uncertainties affecting the conclusion"],
                "suggested_searches": ["5 specific search queries for finding relevant papers"],
                "confidence_assessment": {
                    "level": "High/Medium/Low",
                    "explanation": "Explanation of confidence level"
                }
            }
        """).strip()

        prompt = dedent(f"""
            Synthesize the following analyses into a final assessment:
            
            Claim: {claim_text}
            
            Supporting Mechanisms:
            {json.dumps(supporting_mechanisms, indent=2)}
            
            Contradicting Mechanisms:
            {json.dumps(contradicting_mechanisms, indent=2)}
            
            Theoretical Analysis:
            {json.dumps(theoretical_analysis, indent=2)}
            
            Provide a final synthesis and rating.
        """).strip()

        return await self.openai_service.generate_json_async(prompt, system_prompt)

    def _save_report(self, batch_id: str, claim_id: str, report: Dict):
        """Save the screening report to a file"""
        report_dir = os.path.join('saved_jobs', batch_id)
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f"{claim_id}.txt")
        with open(report_file, 'w') as f:
            json.dump({
                "text": report["claim"],
                "status": "processed",
                "additional_info": report,
                "review_type": "llm",
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    @classmethod
    async def process_batch(cls, claims: List[Claim], batch_id: str, loop=None) -> None:
        """Class method to handle all batch processing"""
        if loop is None:
            loop = asyncio.get_event_loop()
            
        logger.info(f"Starting batch processing of {len(claims)} claims")
        
        screener = cls()
        claim_ids = [str(uuid.uuid4())[:8] for _ in claims]
        
        # Create tasks for all claims
        tasks = []
        for i, (claim, claim_id) in enumerate(zip(claims, claim_ids)):
            logger.info(f"Creating task {i+1}/{len(claims)} for claim {claim_id}")
            tasks.append(screener.screen_claim(claim, batch_id, claim_id, loop=loop))
        
        logger.info(f"Created {len(tasks)} tasks, starting gather")
        
        # Process in smaller chunks to avoid overwhelming the API
        chunk_size = 10
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            try:
                # Create coroutines that start at staggered times
                async def process_with_delay(task, delay):
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return await task

                staggered_tasks = [
                    process_with_delay(task, j) 
                    for j, task in enumerate(chunk)
                ]
                
                results = await asyncio.gather(*staggered_tasks, return_exceptions=True)
                # Check for and log any exceptions
                for j, result in enumerate(results):
                    claim_idx = i + j
                    if claim_idx < len(claim_ids):
                        if isinstance(result, Exception):
                            logger.error(f"Error processing claim {claim_ids[claim_idx]}: {str(result)}")
                        else:
                            logger.info(f"Successfully processed claim {claim_ids[claim_idx]}")
                
                # Small delay between chunks to avoid rate limits
                if i + chunk_size < len(tasks):
                    await asyncio.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        logger.info("All tasks completed")
  