from argparse import Namespace
import pickle


class Unpickler(pickle.Unpickler):
    """Uses Namespace to replace any class that pickler cannot find during unpickling."""

    def find_class(self, mod_name, name):
        try:
            return super().find_class(mod_name, name)
        except (ModuleNotFoundError, AttributeError):
            print(
                f"[lazy_unpickler] Using Namespace instead of class {mod_name}.{name} due to missing dependencies."
            )
            return Namespace
