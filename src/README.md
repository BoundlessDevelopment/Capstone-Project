# How to run the files

We are using a custom version of the petting zoo library where we are trying to add our own custom Multi-Particle Environment (MPE)

To install our custom PettingZoo, you will need to follow these steps:
- Make sure that your env does not have pettingzoo installed. You can do it by: ```pip show pettingzoo```
- If it exists you can uninstall it: ```pip uninstall pettingzoo``` (Type 'Y' when it prompts you)
- Once we are sure that it is not present in the env, go to ```cd src/CustomPettingZoo```
- Run ```pip3 install -e ./```