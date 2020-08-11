from base64 import b64encode, b64decode
from copy import deepcopy
from dataclasses import field, fields, MISSING
from marshmallow import fields as mmfields
from typing import Any, Collection, Mapping, Optional, Callable

import numpy as np
import dataclasses_json.core
from dataclasses_json import config as json_config


_has_overridden_dataclasses_json_asdict = False


def ndarray_field(default: Any = MISSING, *, encode_as_base64: bool = False,
                  custom_encoder: Optional[Callable] = None,
                  custom_decoder: Optional[Callable] = None,
                  **json_config_args) -> field:
    """
    A field for a numpy array that, by default, will be encoded to and decoded from a list during
    JSON (de)serialization.
    """

    _override_dataclasses_json_asdict()

    if custom_encoder:
        encoder = custom_encoder
    else:
        encoder = _encode_array_to_base64 if encode_as_base64 else lambda arr: arr.tolist()

    if custom_decoder:
        decoder = custom_decoder
    else:
        decoder = _decode_base64_to_array if encode_as_base64 else np.array

    return field(
        default=default,
        metadata=json_config(
            encoder=encoder,
            decoder=decoder,
            mm_field=mmfields.Str if encode_as_base64 else mmfields.List,
            **json_config_args
        ))


def _encode_array_to_base64(arr: np.ndarray) -> list:
    """ Returns a base64 representation of a numpy array. """
    return [b64encode(arr).decode("utf-8"), str(arr.dtype), *arr.shape]


def _decode_base64_to_array(b64_object: Optional[list]) -> Optional[np.ndarray]:
    """
    Decodes a base64 representation of a numpy array that was encoded using the function
    _encode_array_to_base64.
    """

    if b64_object is None:
        return None

    return np.frombuffer(
        b64decode(b64_object[0]), dtype=b64_object[1]
    ).reshape(tuple(b64_object[2:]))


def _override_dataclasses_json_asdict() -> None:
    """
    Overrides the _asdict function in dataclasses-json to be compatible with the way we use numpy
    arrays.
    """

    global _has_overridden_dataclasses_json_asdict
    if _has_overridden_dataclasses_json_asdict:
        return

    def _asdict(obj, encode_json=False):
        if dataclasses_json.core._is_dataclass_instance(obj):
            overrides = dataclasses_json.core._user_overrides_or_exts(obj)
            result = []
            for field in fields(obj):
                value = getattr(obj, field.name)
                exclude = overrides[field.name].exclude
                # If the exclude predicate returns true, the key should be
                #  excluded from encoding, so skip the rest of the loop
                if exclude and exclude(value):
                    continue

                value = _asdict(value, encode_json=encode_json)
                result.append((field.name, value))

            result = dataclasses_json.core._handle_undefined_parameters_safe(cls=obj, kvs=dict(result),
                                                       usage="to")
            return dataclasses_json.core._encode_overrides(dict(result), overrides,
                                     encode_json=encode_json)
        elif isinstance(obj, Mapping):
            return dict((_asdict(k, encode_json=encode_json),
                         _asdict(v, encode_json=encode_json)) for k, v in
                        obj.items())
        elif isinstance(obj, Collection) and not isinstance(obj, str) \
                and not isinstance(obj, bytes) \
                and (not isinstance(obj, np.ndarray) or obj.dtype == np.object):  # Makes it work!
            return list(_asdict(v, encode_json=encode_json) for v in obj)
        else:
            return deepcopy(obj)

    dataclasses_json.core._asdict = _asdict
    _has_overridden_dataclasses_json_asdict = True
