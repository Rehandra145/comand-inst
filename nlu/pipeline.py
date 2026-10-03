"""
NLU Pipeline — Orkestrasi seluruh komponen NLU.

Pipeline:
  text → intent classifier → slot extractor → command builder → validator → ROS 2 adapter
"""

import json
import logging
from typing import Dict, Any, Optional

from nlu.intent_classifier import (
    BaseIntentClassifier,
    KeywordIntentClassifier,
)
from nlu.slot_extractor import (
    BaseSlotExtractor,
    RuleBasedSlotExtractor,
)
from nlu.command_builder import build_command
from nlu.validator import validate_command
from nlu.ros2_adapter import BaseROS2Adapter, MockROS2Adapter


logger = logging.getLogger(__name__)


class NLUPipeline:
    """
    Pipeline utama NLU:
      text → intent → slots → command → validate → ROS 2

    Setiap komponen bisa diganti (dependency injection).
    """

    def __init__(
        self,
        intent_classifier: Optional[BaseIntentClassifier] = None,
        slot_extractor: Optional[BaseSlotExtractor] = None,
        ros2_adapter: Optional[BaseROS2Adapter] = None,
    ):
        self.intent_classifier = intent_classifier or KeywordIntentClassifier()
        self.slot_extractor = slot_extractor or RuleBasedSlotExtractor()
        self.ros2_adapter = ros2_adapter or MockROS2Adapter()

    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text melalui seluruh pipeline.

        Returns structured command dict dengan status validasi.
        """

        logger.info(f"[INPUT] text = \"{text}\"")

        # 1. Intent Classification
        intent_result = self.intent_classifier.classify(text)
        logger.info(
            f"[INTENT] intent = {intent_result.intent}, "
            f"confidence = {intent_result.confidence:.2f}"
        )

        # 2. Slot Extraction
        slots = self.slot_extractor.extract(text, intent_result.intent)
        logger.info(f"[SLOTS] {slots if slots else '(none)'}")

        # 3. Command Building
        command = build_command(intent_result, slots)
        logger.info(f"[COMMAND] {json.dumps(command, ensure_ascii=False)}")

        # 4. Validation
        validated = validate_command(command)
        logger.info(
            f"[VALIDATION] status = {validated['status']}"
        )

        if validated.get("missing_slots"):
            logger.warning(
                f"[VALIDATION] missing_slots = {validated['missing_slots']}"
            )

        # 5. ROS 2 Adapter
        if validated["status"] == "VALID":
            success = self.ros2_adapter.send_command(validated)
            validated["ros2_sent"] = success
        else:
            logger.info(
                f"[ROS2] Command tidak dikirim (status={validated['status']})"
            )
            validated["ros2_sent"] = False

        return validated
