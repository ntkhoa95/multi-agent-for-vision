import logging
from typing import Dict, List, Optional, Set, Tuple

import spacy

spacy.load("en_core_web_sm")
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


class NLPProcessor:
    def __init__(self):
        """Initialize NLP processor with spaCy"""
        self.nlp = spacy.load("en_core_web_sm")

        # Existing filter words
        self.filter_words = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "with",
            "and",
            "or",
            "some",
            "any",
            "many",
            "few",
            "all",
            "every",
            "each",
            "this",
            "that",
            "these",
            "those",
            "image",
            "video",
            "picture",
            "frame",
            "scene",
            "view",
        }

        # Add general detection keywords
        self.general_detection_keywords = {
            "objects",
            "things",
            "everything",
            "anything",
            "something",
            "all objects",
            "all things",
        }

        # Existing task verbs
        self.task_verbs = {
            "detection": {
                "detect",
                "find",
                "locate",
                "spot",
                "identify",
                "show",
                "discover",
                "recognize",
                "see",
                "search",
                "look",
            },
            "classification": {
                "classify",
                "categorize",
                "label",
                "name",
                "determine",
                "what",
                "tell",
                "describe",
                "explain",
                "identify",
            },
        }

        # Your existing object synonyms
        self.object_synonyms = {
            "person": {
                "person",
                "people",
                "human",
                "humans",
                "pedestrian",
                "pedestrians",
                "individual",
                "individuals",
                "man",
                "men",
                "woman",
                "women",
            },
            "cat": {"cat", "cats", "kitten", "kittens"},
            "dog": {"dog", "dogs", "puppy", "puppies"},
            "car": {"car", "cars", "vehicle", "vehicles", "automobile"},
            "truck": {"truck", "trucks"},
        }

    def preprocess_query(self, query: str) -> str:
        """Preprocess the query text"""
        query = query.lower()
        query = " ".join(word if "-" in word else word.strip(".,!?();:") for word in query.split())
        return query

    def normalize_object_name(self, name: str) -> str:
        """Normalize object names (e.g., plurals, synonyms)"""
        name = name.lower()

        # Remove plural 's'
        if name.endswith("s"):
            singular = name[:-1]
            # Check if singular form is in any synonym set
            for main_term, synonyms in self.object_synonyms.items():
                if singular in synonyms:
                    return main_term

        # Check direct matches and synonyms
        for main_term, synonyms in self.object_synonyms.items():
            if name in synonyms:
                return main_term

        return name

    def extract_task_type(self, doc: Doc) -> Optional[str]:
        """Extract the main task type from the parsed query"""
        # Check for classification indicators
        if any(token.text.lower() in ["what", "classify", "categorize"] for token in doc):
            return "classification"

        # Check other task verbs
        for token in doc:
            if token.pos_ in ["VERB", "NOUN"]:
                word = token.lemma_.lower()
                for task, verbs in self.task_verbs.items():
                    if word in verbs:
                        return task

        # Default to detection for object-focused queries
        return "detection"

    def extract_target_objects(self, doc: Doc) -> List[str]:
        """Extract target objects from the parsed query using dependency parsing"""
        # Check for general detection keywords first
        query_text = doc.text.lower()
        if any(keyword in query_text for keyword in self.general_detection_keywords):
            return []  # Return empty list for general detection

        target_objects = set()

        # Your existing object extraction logic
        for token in doc:
            if (token.dep_ == "dobj" and token.head.lemma_ in self.task_verbs["detection"]) or (
                token.dep_ == "conj" and token.head.dep_ == "dobj"
            ):
                if token.text.lower() not in self.filter_words:
                    target_objects.add(token.text.lower())

        # If no direct objects found, look for nouns
        if not target_objects:
            for token in doc:
                if token.pos_ == "NOUN" and token.text.lower() not in self.filter_words:
                    target_objects.add(token.text.lower())

        # Normalize object names
        normalized_objects = [self.normalize_object_name(obj) for obj in target_objects]

        return normalized_objects

    def parse_query(self, query: str) -> Tuple[Optional[str], List[str]]:
        """Parse the query to extract task type and target objects"""
        processed_query = self.preprocess_query(query)

        # Check for general detection first
        if any(keyword in processed_query for keyword in self.general_detection_keywords):
            logger.debug("Detected general object detection query")
            return "detection", []

        doc = self.nlp(processed_query)
        task_type = self.extract_task_type(doc)
        target_objects = self.extract_target_objects(doc)

        return task_type, target_objects

    def validate_objects(self, objects: List[str], available_classes: Set[str]) -> List[str]:
        """Validate extracted objects against available model classes"""
        # If no objects specified, return None for general detection
        if not objects:
            return []

        valid_objects = []
        for obj in objects:
            if obj in available_classes:
                valid_objects.append(obj)
            else:
                logger.warning(f"Object '{obj}' not in available classes, will be ignored")
        return valid_objects
