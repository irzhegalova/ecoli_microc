for id in {242..364};
do wget https://regulondb.ccg.unam.mx/wdps/RHTECOLIBSD00${id}/authorData/cvs -O data/chipseq/RHTECOLIBSD00${id}.csv --no-check-certificate; done