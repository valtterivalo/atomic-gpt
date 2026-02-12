CC = gcc
CFLAGS = -O2 -Wall -Wextra
LDFLAGS = -lm

.PHONY: all clean run-c run-py benchmark data

all: gpt

gpt: gpt.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

data: input.txt

input.txt:
	curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt

run-c: gpt input.txt
	./gpt

run-py: input.txt
	python3 gpt.py

benchmark: gpt input.txt
	@bash benchmark.sh

clean:
	rm -f gpt
