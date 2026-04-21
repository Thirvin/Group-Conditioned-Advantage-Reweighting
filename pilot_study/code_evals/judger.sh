#!/usr/bin/env bash

PREFIX="$1"
no="$2"

cd "${PREFIX}" || exit 1

if ! g++ -O2 -std=gnu++2a "temp_${no}.cpp" -o "temp_${no}.out"; then
    echo -1
    exit 0
fi

total=$(ls *.in -1 | wc -l)

for ((i=total-1; i>=0; --i)); do
    if [[ ! -r "${i}.in" || ! -r "${i}.out" ]]; then
        echo -1
        exit 2
    fi

    (
        ulimit -t 2
        ulimit -v 1048576
        ./temp_${no}.out < "${i}.in" > "temp_${no}_${i}.tmp"
    )

    if ! diff -b -q "temp_${no}_${i}.tmp" "${i}.out" > /dev/null; then
        rm -f "temp_${no}_${i}.tmp" "temp_${no}.out"
        echo 0
        exit 0
    fi
    rm -f "temp_${no}_${i}.tmp"
done

rm -f "temp_${no}.out"

echo 1
