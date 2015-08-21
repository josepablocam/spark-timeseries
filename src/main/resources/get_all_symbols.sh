 # Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
 #
 # Cloudera, Inc. licenses this file to you under the Apache License,
 # Version 2.0 (the "License"). You may not use this file except in
 # compliance with the License. You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 # CONDITIONS OF ANY KIND, either express or implied. See the License for
 # the specific language governing permissions and limitations under the
 # License.



# A very simple script to download the data needed for portfolio optimization example.
# If we have the .zip folder from the repo, then just inflate. Otherwise download

FOLDER=data

if [ ! -f ./${FOLDER}.zip ]
then
    mkdir -p ${FOLDER}
    while read SYMBOL; do
      ./get_symbol.sh ${FOLDER} ${SYMBOL}
      sleep 1
    done < symbols.txt
    # remove any downloads that failed and created empty files
    find ${FOLDER} -size 0 | xargs -I {} rm -rf {}
    # go ahead and create a zip for next time
    zip -r ${FOLDER}.zip ${FOLDER}
else
# we have the zipped filed
    unzip ${FOLDER}.zip
fi
