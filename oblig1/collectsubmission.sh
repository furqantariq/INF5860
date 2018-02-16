rm -f INF5860_Oblig1.zip
zip -r INF5860_Oblig1.zip . -x  "*inf5860/datasets/cifar-10-batches-py*" "*.ipynb_checkpoints*"  "*collectSubmission.sh*"  ".env/*" "*.pyc" "*code/build/*"
