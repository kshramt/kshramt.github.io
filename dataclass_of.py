# dataclass_fromdict
# GPL v3
import collections
import dataclasses
import sys
import typing


_PY37 = sys.version_info.major == 3 and sys.version_info.minor == 7


if _PY37:

    def dataclass_of(cls, x):
        if dataclasses.is_dataclass(cls):
            if not isinstance(x, dict):
                raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
            fields = {f.name: f.type for f in dataclasses.fields(cls)}
            if set(fields.keys()) != set(x.keys()):
                raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
            return cls(**{k: dataclass_of(fields[k], v) for k, v in x.items()})
        elif type(cls) == type:
            if not isinstance(x, cls):
                raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
            return x
        elif isinstance(cls, tuple):
            if x not in cls:
                raise TypeError(f"{x} is not compatible with {cls}")
            return x
        elif cls.__origin__ == list or cls.__origin__ == collections.abc.Sequence:
            vcls = cls.__args__[0]
            return [dataclass_of(vcls, v) for v in x]
        elif cls.__origin__ == dict:
            kcls, vcls = cls.__args__
            return {dataclass_of(kcls, k): dataclass_of(vcls, v) for k, v in x.items()}
        elif cls.__origin__ == typing.Union:
            for ucls in cls.__args__:
                try:
                    return dataclass_of(ucls, x)
                except TypeError:
                    pass
            raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
        else:
            raise ValueError(f"Unsupported value {x}: {type(x)}")


else:

    def dataclass_of(cls, x):
        if dataclasses.is_dataclass(cls):
            if not isinstance(x, dict):
                raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
            fields = {f.name: f.type for f in dataclasses.fields(cls)}
            if set(fields.keys()) != set(x.keys()):
                raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
            return cls(**{k: dataclass_of(fields[k], v) for k, v in x.items()})
        elif type(cls) == type:
            if not isinstance(x, cls):
                raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
            return x
        elif cls.__origin__ == typing.Literal:
            if x not in cls.__args__:
                raise TypeError(f"{x} is not compatible with {cls}")
            return x
        elif cls.__origin__ == list or cls.__origin__ == collections.abc.Sequence:
            vcls = cls.__args__[0]
            return [dataclass_of(vcls, v) for v in x]
        elif cls.__origin__ == dict:
            kcls, vcls = cls.__args__
            return {dataclass_of(kcls, k): dataclass_of(vcls, v) for k, v in x.items()}
        elif cls.__origin__ == typing.Union:
            for ucls in cls.__args__:
                try:
                    return dataclass_of(ucls, x)
                except TypeError:
                    pass
            raise TypeError(f"{x}: {type(x)} is not compatible with {cls}")
        else:
            raise ValueError(f"Unsupported value {x}: {type(x)}")


if __name__ == "__main__":
    if _PY37:

        @dataclasses.dataclass
        class c4:
            x: int
            y: float

        @dataclasses.dataclass
        class c3:
            x: ("yy",)
            y: typing.Dict[str, typing.Optional[c4]]

        @dataclasses.dataclass
        class c2:
            x: ("xx",)
            y: typing.Dict[str, typing.Optional[c4]]

        @dataclasses.dataclass
        class c1:
            x: typing.List[typing.Union[c2, c3]]
            y: c4
            z: typing.Sequence[int]

        x = c1(
            x=[c2(x="xx", y=dict(a=None, b=c4(x=2, y=1.0))), c3(x="yy", y=dict())],
            y=c4(x=1, y=1.3),
            z=[1],
        )
        assert x == dataclass_of(c1, dataclasses.asdict(x))
    else:

        @dataclasses.dataclass
        class c4:
            x: int
            y: float

        @dataclasses.dataclass
        class c3:
            x: typing.Literal["yy"]
            y: typing.Dict[str, typing.Optional[c4]]

        @dataclasses.dataclass
        class c2:
            x: typing.Literal["xx", "zz"]
            y: typing.Dict[str, typing.Optional[c4]]

        @dataclasses.dataclass
        class c1:
            x: typing.List[typing.Union[c2, c3]]
            y: c4
            z: typing.Sequence[int]

        x = c1(
            x=[c2(x="xx", y=dict(a=None, b=c4(x=2, y=1.0))), c3(x="yy", y=dict())],
            y=c4(x=1, y=1.3),
            z=[1],
        )
        assert x == dataclass_of(c1, dataclasses.asdict(x))
