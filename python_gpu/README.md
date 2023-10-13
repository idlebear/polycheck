# Polycheck GPU

There is an interesting library out there called [PyCuda](https://documen.tician.de/pycuda) that allows more or less direct access to compiled GPU code instead of going through the C++ interface via PyBind11. This folder contains a polycheck implementation that takes advantage of that capability.  The code is cleaner, easier to maintain and easier to understand.  Sounds like a good idea to me.


