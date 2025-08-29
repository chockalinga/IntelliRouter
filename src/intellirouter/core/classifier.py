"""Prompt classification for intelligent routing."""

import re
import time
from typing import List, Dict, Any, Optional
import tiktoken

from ..models.chat import ChatMessage
from ..models.providers import TaskType
from ..models.routing import (
    PromptClassification,
    ComplexityLevel,
    LatencyRequirement,
)


class PromptClassifier:
    """Classifies prompts to determine routing strategy."""
    
    def __init__(self):
        """Initialize the prompt classifier."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
        # Keywords for task type detection
        self.task_keywords = {
            TaskType.CODING: [
                "code", "function", "class", "method", "algorithm", "debug", "bug",
                "python", "javascript", "java", "c++", "sql", "html", "css", "react",
                "programming", "syntax", "compile", "error", "exception", "import",
                "library", "framework", "api", "database", "query", "script"
            ],
            TaskType.REASONING: [
                "analyze", "explain why", "reason", "logic", "because", "therefore",
                "conclude", "infer", "deduce", "prove", "argument", "evidence",
                "hypothesis", "theory", "principle", "cause", "effect", "problem solving",
                "step by step", "think through", "evaluate", "assess", "compare"
            ],
            TaskType.CREATIVE_WRITING: [
                "story", "poem", "creative", "imagine", "character", "plot", "write",
                "narrative", "fiction", "novel", "short story", "dialogue", "scene",
                "poetry", "verse", "rhyme", "metaphor", "creative writing", "author"
            ],
            TaskType.TRANSLATION: [
                "translate", "translation", "language", "french", "spanish", "german",
                "chinese", "japanese", "korean", "italian", "portuguese", "russian",
                "arabic", "hindi", "from english", "to english", "in spanish"
            ],
            TaskType.SUMMARIZATION: [
                "summarize", "summary", "brief", "overview", "key points", "main ideas",
                "tldr", "abstract", "synopsis", "outline", "highlights", "essence"
            ],
            TaskType.STRUCTURED_OUTPUT: [
                "json", "xml", "yaml", "csv", "table", "list", "format", "structure",
                "schema", "template", "form", "fields", "parse", "extract data"
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            "high_complexity": [
                "complex", "complicated", "advanced", "sophisticated", "intricate",
                "multi-step", "comprehensive", "detailed analysis", "in-depth",
                "expert level", "professional", "technical specification"
            ],
            "low_complexity": [
                "simple", "basic", "easy", "quick", "brief", "straightforward",
                "one-liner", "yes/no", "true/false", "short answer"
            ]
        }
        
        # Latency requirement indicators
        self.latency_indicators = {
            LatencyRequirement.REAL_TIME: [
                "urgent", "asap", "immediately", "quick", "fast", "real-time",
                "instant", "now", "emergency"
            ],
            LatencyRequirement.SLOW: [
                "detailed", "comprehensive", "thorough", "complete analysis",
                "take your time", "no rush", "when convenient"
            ]
        }
    
    def classify(self, messages: List[ChatMessage]) -> PromptClassification:
        """Classify a conversation to determine routing strategy."""
        start_time = time.time()
        
        # Combine all message content
        combined_text = self._combine_messages(messages)
        
        # Basic analysis
        task_type = self._detect_task_type(combined_text, messages)
        complexity_level = self._assess_complexity(combined_text, messages)
        estimated_tokens = self._estimate_tokens(combined_text)
        
        # Capability requirements
        requires_tools = self._check_tools_requirement(messages)
        requires_json_mode = self._check_json_requirement(combined_text)
        requires_vision = self._check_vision_requirement(messages)
        
        # Content analysis
        has_code = self._detect_code(combined_text)
        has_math = self._detect_math(combined_text)
        has_structured_data = self._detect_structured_data(combined_text)
        is_conversational = self._detect_conversational(messages)
        
        # Inferred requirements
        latency_requirement = self._infer_latency_requirement(combined_text, complexity_level)
        quality_threshold = self._determine_quality_threshold(task_type, complexity_level)
        
        # Classification confidence
        confidence = self._calculate_confidence(task_type, complexity_level, combined_text)
        
        classification_time = int((time.time() - start_time) * 1000)
        
        return PromptClassification(
            task_type=task_type,
            complexity_level=complexity_level,
            estimated_tokens=estimated_tokens,
            requires_tools=requires_tools,
            requires_json_mode=requires_json_mode,
            requires_vision=requires_vision,
            has_code=has_code,
            has_math=has_math,
            has_structured_data=has_structured_data,
            is_conversational=is_conversational,
            latency_requirement=latency_requirement,
            quality_threshold=quality_threshold,
            confidence=confidence,
            classification_time_ms=classification_time
        )
    
    def _combine_messages(self, messages: List[ChatMessage]) -> str:
        """Combine all message content into a single string."""
        combined = []
        for message in messages:
            if message.content:
                combined.append(message.content)
        return " ".join(combined).lower()
    
    def _detect_task_type(self, text: str, messages: List[ChatMessage]) -> TaskType:
        """Detect the primary task type from the text."""
        task_scores = {}
        
        for task_type, keywords in self.task_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += text.count(keyword)
            task_scores[task_type] = score
        
        # Special cases and heuristics
        
        # Check for code blocks
        if "```" in "".join([msg.content or "" for msg in messages]):
            task_scores[TaskType.CODING] = task_scores.get(TaskType.CODING, 0) + 10
        
        # Check for mathematical notation
        if re.search(r'[∫∑∆∇αβγθλμπσΩ]|\\[a-zA-Z]+|\$.*?\$', text):
            task_scores[TaskType.REASONING] = task_scores.get(TaskType.REASONING, 0) + 5
        
        # Check for question answering patterns
        if any(q in text for q in ["what is", "how do", "can you explain", "tell me about"]):
            if not task_scores or max(task_scores.values()) < 3:
                return TaskType.CHAT
        
        # Return the task type with highest score, default to CHAT
        if task_scores:
            return max(task_scores, key=task_scores.get)
        return TaskType.CHAT
    
    def _assess_complexity(self, text: str, messages: List[ChatMessage]) -> ComplexityLevel:
        """Assess the complexity level of the request."""
        complexity_score = 0
        
        # Check complexity indicators
        for indicator in self.complexity_indicators["high_complexity"]:
            if indicator in text:
                complexity_score += 2
        
        for indicator in self.complexity_indicators["low_complexity"]:
            if indicator in text:
                complexity_score -= 1
        
        # Length-based complexity
        token_count = self._estimate_tokens(text)
        if token_count > 2000:
            complexity_score += 3
        elif token_count > 1000:
            complexity_score += 2
        elif token_count > 500:
            complexity_score += 1
        elif token_count < 50:
            complexity_score -= 2
        
        # Number of messages (conversation complexity)
        if len(messages) > 10:
            complexity_score += 2
        elif len(messages) > 5:
            complexity_score += 1
        
        # Code complexity
        if "```" in text:
            complexity_score += 1
        
        # Multi-part questions
        question_count = text.count("?")
        if question_count > 3:
            complexity_score += 2
        elif question_count > 1:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 5:
            return ComplexityLevel.VERY_HIGH
        elif complexity_score >= 3:
            return ComplexityLevel.HIGH
        elif complexity_score >= 1:
            return ComplexityLevel.MEDIUM
        elif complexity_score >= -1:
            return ComplexityLevel.LOW
        else:
            return ComplexityLevel.VERY_LOW
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for the text."""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback estimation: ~4 characters per token
            return len(text) // 4
    
    def _check_tools_requirement(self, messages: List[ChatMessage]) -> bool:
        """Check if the request requires tool/function calling."""
        # Check if tools are already present in the request
        for message in messages:
            if message.tool_calls:
                return True
        
        # Check for tool-related keywords in content
        tool_keywords = [
            "call function", "use tool", "execute", "run", "calculate",
            "get data", "fetch", "search", "lookup", "api call"
        ]
        
        combined_text = self._combine_messages(messages)
        return any(keyword in combined_text for keyword in tool_keywords)
    
    def _check_json_requirement(self, text: str) -> bool:
        """Check if the request requires JSON mode."""
        json_indicators = [
            "json", "structured data", "format as json", "return json",
            "schema", "object", "array", "key-value", "parse"
        ]
        return any(indicator in text for indicator in json_indicators)
    
    def _check_vision_requirement(self, messages: List[ChatMessage]) -> bool:
        """Check if the request involves vision/image processing."""
        vision_keywords = [
            "image", "picture", "photo", "visual", "see", "look at",
            "analyze image", "describe picture", "what's in", "identify"
        ]
        
        combined_text = self._combine_messages(messages)
        return any(keyword in combined_text for keyword in vision_keywords)
    
    def _detect_code(self, text: str) -> bool:
        """Detect if the text contains code."""
        code_indicators = [
            "```", "def ", "class ", "function", "import ", "from ",
            "if __name__", "return ", "print(", "console.log"
        ]
        return any(indicator in text for indicator in code_indicators)
    
    def _detect_math(self, text: str) -> bool:
        """Detect mathematical content."""
        math_patterns = [
            r'\d+\s*[+\-*/]\s*\d+',  # Basic arithmetic
            r'[∫∑∆∇αβγθλμπσΩ]',      # Mathematical symbols
            r'\\[a-zA-Z]+',           # LaTeX commands
            r'\$.*?\$',               # LaTeX math mode
            r'equation', r'formula', r'calculate', r'solve'
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _detect_structured_data(self, text: str) -> bool:
        """Detect structured data in the text."""
        structured_indicators = [
            "{", "}", "[", "]", ":", "csv", "table", "rows", "columns"
        ]
        return any(indicator in text for indicator in structured_indicators)
    
    def _detect_conversational(self, messages: List[ChatMessage]) -> bool:
        """Detect if this is part of a conversation."""
        if len(messages) > 2:
            return True
        
        conversational_indicators = [
            "continue", "also", "and", "but", "however", "furthermore",
            "as we discussed", "like i mentioned", "following up"
        ]
        
        combined_text = self._combine_messages(messages)
        return any(indicator in combined_text for indicator in conversational_indicators)
    
    def _infer_latency_requirement(
        self, 
        text: str, 
        complexity: ComplexityLevel
    ) -> LatencyRequirement:
        """Infer latency requirements from text and complexity."""
        
        # Check explicit latency indicators
        for latency_req, indicators in self.latency_indicators.items():
            if any(indicator in text for indicator in indicators):
                return latency_req
        
        # Infer from complexity
        if complexity in [ComplexityLevel.VERY_HIGH, ComplexityLevel.HIGH]:
            return LatencyRequirement.SLOW
        elif complexity == ComplexityLevel.VERY_LOW:
            return LatencyRequirement.FAST
        else:
            return LatencyRequirement.NORMAL
    
    def _determine_quality_threshold(
        self, 
        task_type: TaskType, 
        complexity: ComplexityLevel
    ) -> float:
        """Determine minimum quality threshold based on task and complexity."""
        
        # Base thresholds by task type
        task_thresholds = {
            TaskType.CODING: 0.85,
            TaskType.REASONING: 0.90,
            TaskType.CREATIVE_WRITING: 0.75,
            TaskType.TRANSLATION: 0.80,
            TaskType.SUMMARIZATION: 0.70,
            TaskType.STRUCTURED_OUTPUT: 0.85,
            TaskType.RAG: 0.80,
            TaskType.CHAT: 0.60,
        }
        
        base_threshold = task_thresholds.get(task_type, 0.70)
        
        # Adjust based on complexity
        complexity_adjustments = {
            ComplexityLevel.VERY_HIGH: 0.10,
            ComplexityLevel.HIGH: 0.05,
            ComplexityLevel.MEDIUM: 0.0,
            ComplexityLevel.LOW: -0.05,
            ComplexityLevel.VERY_LOW: -0.10,
        }
        
        adjustment = complexity_adjustments.get(complexity, 0.0)
        
        # Ensure threshold is within valid range
        return max(0.0, min(1.0, base_threshold + adjustment))
    
    def _calculate_confidence(
        self, 
        task_type: TaskType, 
        complexity: ComplexityLevel, 
        text: str
    ) -> float:
        """Calculate confidence in the classification."""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on clear indicators
        task_keywords = self.task_keywords.get(task_type, [])
        keyword_matches = sum(1 for keyword in task_keywords if keyword in text)
        
        if keyword_matches > 0:
            confidence += 0.1 * min(keyword_matches, 3)  # Max +0.3
        
        # Adjust based on text length (more text = more confidence)
        token_count = self._estimate_tokens(text)
        if token_count > 100:
            confidence += 0.1
        if token_count > 500:
            confidence += 0.1
        
        # Complexity consistency check
        if complexity in [ComplexityLevel.VERY_HIGH, ComplexityLevel.VERY_LOW]:
            confidence += 0.1  # More confident about extreme classifications
        
        return min(1.0, confidence)
