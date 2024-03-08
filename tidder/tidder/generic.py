from typing import (
    ClassVar,
    Generic,
    Type,
    TypeVar,
)
import abc
# Generic type for creating dependencies between Processors and results.
# NOTE: One can bound the type to a specific class, e.g.: TypeVar("T", bound=Video)
# This would allow to access the attributes of the class without having to cast.
T = TypeVar("T")

# NOTE: Generic types cannot be used to ensure that a class inherits from multiple other
# classes. Meaning that natively generic types cannot handle mixins. To workaround this
# we should probably use Protocols. This would introduce some code duplication as we would
# need to define empty interfaces (protocols) for mixing functionality in abstract classes.
# If we don't create the empty interfaces, and use protocols as abstract classes,
# type checking can be compromised because classes that do not fully implement the
# protocol but do inherit from it, are still considered an instance of that
# protocol.


class InstanceableGeneric(Generic[T], abc.ABC):
    """Instanceable generic type.

    Attributes
    ----------
    GenericType : type T or None
        Generic Type.
    """

    GenericType: ClassVar[Type[T] | None] = None

    def __class_getitem__(cls, key):
        r"""Extract Generic type and save in GenericType attribute.

        This is a convenience method that takes the specialization of the generic type
        passed to the Generic and saves it in the GenericType attribute.

        Thus, this method acts as a bridge between type annotation and actual class
        objects. This is mainly useful when defining Generic methods that use methods
        from the generic table class.

        It is important to note that this function is not required for working with
        Generics and it can be bypassed by passing `table_type` during subclass
        creation. For more information see `__init_subclass__`.

        Before modifying anything related to this function, make sure to
        experiment and understand the behaviour of this functionality.

        Reference:
        - How to get Generic classes at runtime:
            https://stackoverflow.com/a/69129940/7454638
        - __class_getitem__ documentation:
            https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem
        """
        # There are two instances where type(cls.GenericType) is not TypeVar:
        # 1. cls.GenericType is a subscriptable type (i.e.: Sequence).
        # 2. A specialized type was already set by another subclass.
        #    This occurs whenever a subclass of Generic that also defines a generic
        #    type (i.e.: SpecializedGeneric) is subclassed more than once.
        if cls.GenericType is None or type(cls.GenericType) is TypeVar:
            cls.GenericType = key
        else:
            try:
                cls.GenericType = cls.GenericType[key]
            except TypeError:
                cls.GenericType = key
        return super().__class_getitem__(key)

    def __init_subclass__(cls, *, table_type: Type[T] = None, **kwargs):
        r"""Set GenericType attribute at subclass creation.

        As defined at PEP-487, this function is a hook method that is triggered whenever
        the parent class is being subclassed.

        Specifically for Generic classes, this method is responsible for setting the
        specialization of the generic table type in the GenericType attribute. The
        specialization can come from the typing annotation, implicitly populated by
        __class_getitem__. Or explicitly set during class creation:

        ```python
        class T:
            pass

        # If typing annotations are used, the correct GenericType is extracted from the
        # specialization of the generic table type
        class TypedGeneric(Generic[T]):
            pass

        # If typing annotations are not used, one can set the GenericType as:
        class UntypedGeneric(Generic, table_type = T):
            pass
        ```

        Before modifying anything related to this function, make sure to
        experiment and understand the behaviour of this functionality.

        Reference:
        - Understanding __init_subclass__:
            https://stackoverflow.com/q/45400284/7454638
        - PEP-487
            https://peps.python.org/pep-0487/
        - __init_subclass__ documentation
            https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
        """
        super().__init_subclass__(**kwargs)

        # Only use Generic.GenericType (probably populated by __class_getitem__)
        # if table_type was not explicitly set
        if table_type is None and InstanceableGeneric.GenericType is not None:
            table_type = InstanceableGeneric.GenericType

        # Always reset Generic.GenericType
        InstanceableGeneric.GenericType = None

        # Only set cls.GenericType if it was explicitly passed or correctly inferred.
        if table_type is not None:
            cls.GenericType = table_type