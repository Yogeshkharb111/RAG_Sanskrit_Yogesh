from rag_pipeline import build_pipeline_index, answer_query

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--query', type=str)
    args = parser.parse_args()

    if args.build:
        build_pipeline_index()
    elif args.query:
        print(answer_query(args.query))
    else:
        print('Usage: main.py --build OR main.py --query "question"')
