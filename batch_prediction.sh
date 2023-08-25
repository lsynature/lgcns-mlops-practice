BASH_ENV=~/.bashrc
ROOT_PATH=/workspaces/lgcns-mlops-practice
PIPENV_PIPFILE=$ROOT_PATH/Pipfile

export PATH=$PATH:/usr/local/py-utils/bin
export PIPENV_PIPFILE=$PIPENV_PIPFILE
pipenv run python $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1
