for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do 
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/${fname}.gz
    fi
done    
