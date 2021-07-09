Plot SWC file
=============

import .swc and export other extension file (e.g., .svg, .pdf, .png) using cli

Requirement
-----------

* python > 3.7
* numpy
* matplotlib

Use example
-----------

```
usage: main.py [-h] [-o OUTPUT] [-r] [--show] FILE

positional arguments:
  FILE                  swc file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output picture name
  -r, --use-radius      plot radius
  --show                show preview
```

### Example

load `neuron.swc` and output `neuron.png` file.

```bash
python main.py neuron.swc
```

output `neuron.svg` format file

```bash
python main.py neuron.swc -o neuron.svg
```

show preview window

```bash
python main.py neuron.swc --show
```

add radius property

```bash
python main.py neuron.swc -r
```