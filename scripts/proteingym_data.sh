VERSION="v1.3"
FILENAME="DMS_ProteinGym_substitutions.zip"
curl -o $SCRATCH/proteingym/data/${FILENAME} https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}/${FILENAME}
unzip $SCRATCH/proteingym/data/${FILENAME} -d $SCRATCH/proteingym/data/ && rm $SCRATCH/proteingym/data/${FILENAME}