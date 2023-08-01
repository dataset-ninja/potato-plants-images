Dataset **Potato Plants Images** can be downloaded in Supervisely format:

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/f/h/02/H8lXlzwrbM31Kzv69G94qAhbKtFK3QOtt4WkmTudCOegmeEyDsb7LueSkMGkpREuOFllzPBp49Ray0jCKkO83Aq7WaYho0LS8RHCuc4R7qJ7k4W7HyczCX8QuPEp.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Potato Plants Images', dst_dir='~/dataset-ninja/')
```
The data in original format can be 🔗[downloaded here](https://www.webpages.uidaho.edu/vakanski/Multispectral_Images_Dataset.html)