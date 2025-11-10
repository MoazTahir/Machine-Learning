"""Dynamic model registry that wires algorithm modules into the FastAPI app."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class RegistryEntry:
    slug: str
    display_name: str
    task_type: str
    module_path: Path
    factory_name: str = "get_service"
    request_model_name: str = "RequestModel"
    response_model_name: str = "ResponseModel"

    def load_module(self) -> ModuleType:
        return _load_module_cached(self.slug, str(self.module_path))

    def build_service(self) -> Any:
        module = self.load_module()
        factory: Callable[[], Any] = getattr(module, self.factory_name)
        return factory()

    def request_model(self) -> type[BaseModel]:
        module = self.load_module()
        return getattr(module, self.request_model_name)

    def response_model(self) -> type[BaseModel]:
        module = self.load_module()
        return getattr(module, self.response_model_name)

    def register_route(self, router: APIRouter) -> None:
        request_model = self.request_model()
        response_model = self.response_model()
        service = self.build_service()

        async def endpoint(payload: request_model) -> response_model:  # type: ignore[misc]
            result = service.predict(payload)
            if isinstance(result, response_model):
                return result
            return response_model(**result)

        endpoint.__name__ = f"predict_{self.slug}"
        router.post(
            f"/models/{self.slug}",
            response_model=response_model,
            summary=f"Run inference for {self.display_name}",
            tags=[self.task_type],
        )(endpoint)


_REGISTRY: dict[str, RegistryEntry] = {
    "linear_regression": RegistryEntry(
        slug="linear_regression",
        display_name="Linear Regression (Salary Prediction)",
        task_type="regression",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Linear Regression"
        / "src"
        / "inference.py",
    ),
    "logistic_regression": RegistryEntry(
        slug="logistic_regression",
        display_name="Logistic Regression (Heart Disease)",
        task_type="classification",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Logistic Regression"
        / "src"
        / "inference.py",
    ),
    "naive_bayes": RegistryEntry(
        slug="naive_bayes",
        display_name="Naive Bayes (Mushroom Edibility)",
        task_type="classification",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Naive Bayes"
        / "src"
        / "inference.py",
    ),
    "support_vector_machine": RegistryEntry(
        slug="support_vector_machine",
        display_name="Support Vector Machine (Breast Cancer)",
        task_type="classification",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Support Vector Machine"
        / "Classification"
        / "src"
        / "inference.py",
    ),
    "support_vector_regression": RegistryEntry(
        slug="support_vector_regression",
        display_name="Support Vector Regression (California Housing)",
        task_type="regression",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Support Vector Machine"
        / "Regression"
        / "src"
        / "inference.py",
    ),
    "knn_classification": RegistryEntry(
        slug="knn_classification",
        display_name="K-Nearest Neighbours (Wine Classification)",
        task_type="classification",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "K Nearest Neighbours"
        / "Classification"
        / "src"
        / "inference.py",
    ),
    "knn_regression": RegistryEntry(
        slug="knn_regression",
        display_name="K-Nearest Neighbours (Diabetes Regression)",
        task_type="regression",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "K Nearest Neighbours"
        / "Regression"
        / "src"
        / "inference.py",
    ),
    "decision_tree_classification": RegistryEntry(
        slug="decision_tree_classification",
        display_name="Decision Tree (Iris Classification)",
        task_type="classification",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Decision Tree"
        / "Classification"
        / "src"
        / "inference.py",
    ),
    "decision_tree_regression": RegistryEntry(
        slug="decision_tree_regression",
        display_name="Decision Tree (California Housing Regression)",
        task_type="regression",
        module_path=REPO_ROOT
        / "Supervised Learning"
        / "Decision Tree"
        / "Regression"
        / "src"
        / "inference.py",
    ),
}


def get_registry() -> dict[str, RegistryEntry]:
    """Expose the registry mapping. Returns the live dictionary for updates."""
    return _REGISTRY


@lru_cache(maxsize=None)
def _load_module_cached(slug: str, module_path: str) -> ModuleType:
    module_name = f"ml_{slug}_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module for {slug} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
