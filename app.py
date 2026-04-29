import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import networkx as nx

load_dotenv()

MODEL = os.getenv("MODEL", "deepseek-chat")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
app = FastAPI(title="AI Learning Companion")


@dataclass
class KnowledgeNode:
    id: str
    name: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    difficulty: int = 1


@dataclass
class KnowledgeState:
    mastery: float = 0.0
    confidence: float = 0.5
    mistake_count: int = 0
    streak: int = 0
    last_review: str = ""


@dataclass
class LearnerProfile:
    name: str
    learning_style: str = "visual"
    completed_nodes: List[str] = field(default_factory=list)
    current_node: Optional[str] = None
    knowledge_levels: Dict[str, KnowledgeState] = field(default_factory=dict)
    mistake_patterns: Dict[str, int] = field(default_factory=dict)
    quiz_history: List[Dict] = field(default_factory=list)


def build_knowledge_graph():
    nodes = {
        "N01": KnowledgeNode("N01", "有理数运算", "加减乘除与混合运算", [], 1),
        "N02": KnowledgeNode("N02", "整式", "单项式与多项式", ["N01"], 1),
        "N03": KnowledgeNode("N03", "一元一次方程", "解法及应用", ["N02"], 2),
        "N04": KnowledgeNode("N04", "二元一次方程组", "代入与消元", ["N03"], 2),
        "N05": KnowledgeNode("N05", "不等式", "一元一次不等式", ["N03"], 2),
        "N06": KnowledgeNode("N06", "二次根式", "根式运算", ["N02"], 2),
        "N07": KnowledgeNode("N07", "一元二次方程", "求根公式", ["N06", "N04"], 3),
    }

    graph = nx.DiGraph()

    for nid, node in nodes.items():
        graph.add_node(nid, data=node)

    for nid, node in nodes.items():
        for pre in node.prerequisites:
            graph.add_edge(pre, nid)

    return nodes, graph


KG, GRAPH = build_knowledge_graph()


def call_llm(system_prompt: str, user_prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except Exception:
        return {"error": content}


class LearningAgent:
    def __init__(self):
        self.profile = None

    def create_profile(self, name: str):
        profile = LearnerProfile(name=name)

        for nid in KG:
            profile.knowledge_levels[nid] = KnowledgeState()

        self.profile = profile
        return profile

    def adaptive_difficulty(self, node_id: str):
        state = self.profile.knowledge_levels[node_id]

        if state.mastery < 0.3:
            return "easy"
        elif state.mastery < 0.7:
            return "medium"
        return "hard"

    def generate_exercise(self, node_id: str):
        node = KG[node_id]
        difficulty = self.adaptive_difficulty(node_id)

        prompt = f"""
知识点：{node.name}
描述：{node.description}
难度：{difficulty}

返回 JSON:
{{
    "question": "题目",
    "correct_answer": "答案",
    "hint": "提示",
    "explanation": "解析"
}}
"""

        return call_llm("你是一位数学老师", prompt)

    def analyze_mistake(self, question, answer, correct_answer):
        prompt = f"""
题目：{question}
学生答案：{answer}
正确答案：{correct_answer}

错误类型：
- calculation
- concept
- careless
- sign_error
- misunderstanding

返回 JSON:
{{
    "mistake_type": "类型",
    "feedback": "反馈"
}}
"""
        return call_llm("你是一位数学诊断专家", prompt)

    def update_mastery(self, node_id: str, correct: bool):
        state = self.profile.knowledge_levels[node_id]

        if correct:
            state.mastery = min(1.0, state.mastery + 0.15)
            state.confidence = min(1.0, state.confidence + 0.1)
            state.streak += 1
        else:
            state.mastery = max(0.0, state.mastery - 0.08)
            state.confidence = max(0.0, state.confidence - 0.05)
            state.mistake_count += 1
            state.streak = 0

        state.last_review = datetime.now().isoformat()

    def plan_learning_path(self):
        weak_nodes = []

        for nid, state in self.profile.knowledge_levels.items():
            if state.mastery < 0.7:
                weak_nodes.append((nid, state.mastery))

        weak_nodes.sort(key=lambda x: x[1])

        result = []

        for nid, _ in weak_nodes:
            prerequisites_met = True

            for pre in KG[nid].prerequisites:
                if self.profile.knowledge_levels[pre].mastery < 0.6:
                    prerequisites_met = False
                    break

            if prerequisites_met:
                result.append(nid)

        return result


agent = LearningAgent()


class CreateProfileRequest(BaseModel):
    name: str


class ExerciseRequest(BaseModel):
    node_id: str


class SubmitAnswerRequest(BaseModel):
    node_id: str
    question: str
    student_answer: str
    correct_answer: str


@app.get("/")
def home():
    return {"message": "AI Learning Companion Running"}


@app.post("/profile")
def create_profile(req: CreateProfileRequest):
    profile = agent.create_profile(req.name)

    return {
        "message": "profile created",
        "profile": asdict(profile)
    }


@app.get("/path")
def learning_path():
    path = agent.plan_learning_path()

    return {
        "path": [KG[n].name for n in path]
    }


@app.post("/exercise")
def generate_exercise(req: ExerciseRequest):
    return agent.generate_exercise(req.node_id)


@app.post("/submit")
def submit_answer(req: SubmitAnswerRequest):
    correct = req.student_answer.strip() == req.correct_answer.strip()

    agent.update_mastery(req.node_id, correct)

    analysis = agent.analyze_mistake(
        req.question,
        req.student_answer,
        req.correct_answer
    )

    return {
        "correct": correct,
        "analysis": analysis,
        "mastery": agent.profile.knowledge_levels[req.node_id].mastery
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
