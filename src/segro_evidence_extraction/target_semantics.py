"""Deterministic target-intent derivation for retrieval diagnostics."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from segro_evidence_extraction.models.base import StrictBaseModel
from segro_evidence_extraction.models.common import ExpectedDataType
from segro_evidence_extraction.models.target import TargetSpecification
from segro_evidence_extraction.vertical_slice import ValueShapeFamily, infer_value_shape_assignment

SupportClassification = Literal[
    "supports_requested_attribute",
    "component_only",
    "attribute_only",
    "weak_context",
    "conflicting",
    "irrelevant",
]


class TargetIntent(StrictBaseModel):
    target_row_id: str
    field_name: str
    field_definition: str
    domain: str
    sub_domain: str
    value_shape_family: ValueShapeFamily
    primary_component: str
    component_terms: list[str]
    requested_attribute: str
    attribute_terms: list[str]
    qualifiers: list[str] = Field(default_factory=list)
    expected_value_indicators: list[str] = Field(default_factory=list)
    likely_source_types: list[str] = Field(default_factory=list)
    likely_evidence_forms: list[str] = Field(default_factory=list)
    exclusion_terms: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


ATTRIBUTE_SUFFIXES: dict[str, list[str]] = {
    "manufacturer": ["manufacturer", "supplier", "company", "make"],
    "model_name": ["model", "model name", "type", "reference"],
    "model_reference": ["model", "reference", "ref"],
    "serial_number": ["serial", "serial number"],
    "installation_date": ["date", "installed", "installation"],
    "construction_date": ["date", "construction", "completion"],
    "count": ["count", "number", "quantity", "no"],
    "quantity": ["count", "number", "quantity", "no"],
    "area_value": ["area", "floor area", "m2", "sqm"],
    "value": ["value", "number", "measurement"],
    "description": ["description", "type", "system", "construction"],
    "component_type": ["component", "type"],
    "component_sub_type": ["component", "sub type", "subtype"],
    "component_name": ["component", "name"],
    "material_type": ["material", "type"],
    "primary_construction_type": ["construction", "type"],
    "sub_type": ["sub type", "subtype", "type"],
    "type": ["type", "category", "class"],
    "reference": ["reference", "ref", "application"],
}


GENERIC_COMPONENT_STOPWORDS = {
    "component",
    "construction",
    "description",
    "date",
    "value",
    "count",
    "quantity",
    "type",
    "sub",
    "primary",
    "name",
    "model",
    "manufacturer",
    "material",
    "reference",
    "installation",
}


def derive_target_intent(target: TargetSpecification) -> TargetIntent:
    field = target.expected_field.lower()
    value_shape = infer_value_shape_assignment(target).value_shape_family
    attribute, attribute_terms = _attribute_for_field(field, target)
    component = _component_for_field(field, attribute_terms)
    component_terms = _component_terms(component)
    indicators = _value_indicators(target, value_shape, attribute_terms)
    source_types = _source_types(target, component_terms, attribute_terms)
    forms = _evidence_forms(target, value_shape, attribute_terms)
    ambiguity = _ambiguity_notes(target, component, attribute)
    return TargetIntent(
        target_row_id=target.target_row_id,
        field_name=target.expected_field,
        field_definition=target.requirement_text,
        domain=str(target.metadata.get("domain") or target.sub_domain),
        sub_domain=target.sub_domain,
        value_shape_family=value_shape,
        primary_component=component,
        component_terms=component_terms,
        requested_attribute=attribute,
        attribute_terms=attribute_terms,
        qualifiers=_qualifiers(field),
        expected_value_indicators=indicators,
        likely_source_types=source_types,
        likely_evidence_forms=forms,
        exclusion_terms=_exclusion_terms(field),
        ambiguity_notes=ambiguity,
    )


def _attribute_for_field(
    field: str,
    target: TargetSpecification,
) -> tuple[str, list[str]]:
    for suffix, terms in sorted(ATTRIBUTE_SUFFIXES.items(), key=lambda item: -len(item[0])):
        if field.endswith(suffix) or f"_{suffix}_" in field:
            return suffix, terms
    if target.expected_data_type == ExpectedDataType.DATE:
        return "date", ["date", "dated", "completion", "certificate"]
    if target.expected_data_type == ExpectedDataType.INTEGER:
        return "count", ATTRIBUTE_SUFFIXES["count"]
    if target.expected_data_type == ExpectedDataType.DECIMAL:
        return "measurement", ["value", "measurement", "area", "m2", "sqm", "kw", "kwh"]
    if target.expected_data_type == ExpectedDataType.BOOLEAN:
        return "presence", ["present", "provided", "installed", "yes", "no"]
    if target.accepted_values or target.expected_data_type == ExpectedDataType.ENUM:
        return "category", ["category", "class", "type"]
    return "description", ATTRIBUTE_SUFFIXES["description"]


def _component_for_field(field: str, attribute_terms: list[str]) -> str:
    tokens = field.split("_")
    attribute_token_set = {
        token for term in attribute_terms for token in re.findall(r"[a-z0-9]+", term)
    }
    kept = [
        token
        for token in tokens
        if token not in GENERIC_COMPONENT_STOPWORDS and token not in attribute_token_set
    ]
    if not kept:
        kept = [token for token in tokens if token not in attribute_token_set] or tokens[:1]
    return " ".join(kept)


def _component_terms(component: str) -> list[str]:
    terms = [component]
    aliases = {
        "pv": ["photovoltaic", "solar pv", "pv panel"],
        "dock leveller": ["dock leveller", "dock levellers", "loading dock"],
        "fire alarm": ["fire alarm", "fire detection", "alarm system"],
        "floor": ["floor", "slab", "concrete slab"],
        "wall": ["wall", "cladding", "elevation"],
        "roof": ["roof", "roofing"],
        "office": ["office", "offices"],
    }
    for key, values in aliases.items():
        if key in component:
            terms.extend(values)
    return sorted(set(term for term in terms if term))


def _value_indicators(
    target: TargetSpecification,
    value_shape: ValueShapeFamily,
    attribute_terms: list[str],
) -> list[str]:
    indicators = list(attribute_terms)
    if value_shape == "integer_count":
        indicators.extend(["no", "number", "quantity"])
    elif value_shape == "decimal_measurement":
        indicators.extend(["m2", "m²", "sqm", "kw", "kwp", "kwh", "%"])
    elif value_shape == "date":
        indicators.extend(["dated", "date", "completion"])
    elif value_shape == "ordered_or_unordered_list":
        indicators.extend(["class", "classes", "list"])
    if target.unit:
        indicators.append(target.unit.lower())
    return sorted(set(indicators))


def _source_types(
    target: TargetSpecification,
    component_terms: list[str],
    attribute_terms: list[str],
) -> list[str]:
    text = " ".join(
        [target.expected_field, target.requirement_text, *component_terms, *attribute_terms]
    )
    text = text.lower()
    types: list[str] = []
    if any(term in text for term in ["planning", "consent", "use class"]):
        types.append("planning approvals")
    if any(term in text for term in ["certificate", "date", "completion"]):
        types.append("certificates")
    if any(term in text for term in ["fire", "alarm", "service", "electrical"]):
        types.append("building services")
    if any(term in text for term in ["roof", "wall", "floor", "cladding", "dock"]):
        types.append("building fabric/general")
    if any(term in text for term in ["fence", "gate", "yard", "external"]):
        types.append("external works")
    if any(term in text for term in ["pv", "photovoltaic", "solar", "energy"]):
        types.append("appendices/energy")
    return types or ["bounded manuals"]


def _evidence_forms(
    target: TargetSpecification,
    value_shape: ValueShapeFamily,
    attribute_terms: list[str],
) -> list[str]:
    forms = ["heading", "paragraph", "schedule"]
    if value_shape == "date" or "certificate" in attribute_terms:
        forms.append("certificate")
    if value_shape in {"integer_count", "decimal_measurement", "ordered_or_unordered_list"}:
        forms.append("table")
    if target.expected_data_type in {ExpectedDataType.INTEGER, ExpectedDataType.DECIMAL}:
        forms.append("numeric statement")
    return sorted(set(forms))


def _qualifiers(field: str) -> list[str]:
    return [
        token
        for token in ["primary", "external", "landlord", "tenant", "approved"]
        if token in field
    ]


def _exclusion_terms(field: str) -> list[str]:
    if "office_area" in field:
        return ["slab", "yard", "warehouse floor loading"]
    if "manufacturer" in field or "model" in field:
        return ["maintenance", "description only", "generic"]
    return []


def _ambiguity_notes(target: TargetSpecification, component: str, attribute: str) -> list[str]:
    notes: list[str] = []
    if not component or len(component) <= 2:
        notes.append("Component subject is weakly inferred from field name.")
    if target.unit and attribute in {"description", "manufacturer", "model_name"}:
        notes.append("Dictionary unit may not apply to narrative or identifier attribute.")
    if target.expected_data_type == ExpectedDataType.DECIMAL and attribute in {
        "manufacturer",
        "model_name",
        "reference",
    }:
        notes.append("Dictionary datatype appears inconsistent with identifier-like field.")
    return notes
