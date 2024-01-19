# Running Tests

To run the test suite, please perform the following from the root directory of this repository:

1. `pip install -e .[testing]`

      This will install all the testing requirements.
2. `sudo apt-get update; sudo apt-get install git-lfs -y`

      We need git-lfs on our system to run some of the tests

3. `pytest ./tests/`

      We need to set an environmental variable to make sure the private API tests can run.
