This directory contains data files that will be loaded by the software, with a subdirectory for each
type of data.


presets
-------
Contains presets easily can be loaded from the program. Presets that are applied to all sections are
located in the subdirectory "run_instance", while the directories "detector_properties",
"fluorophore_settings", "imaging_system_settings", "pulse_scheme", and "sample_properties" contain
presets for their respective sections. A preset file is a JSON file in the exact same format as a
user-saved configuration file, except that it has an additional field "name", which corresponds to
the name of the preset that should be displayed in the program.


sample_structures
-----------------
Contains sample structure data. Each sample structure type has its own folder that contains two
files; "image.tiff" and "properties.json", which contain an image of the sample and its structural
properties respectively.
