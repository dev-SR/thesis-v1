# from __future__ import print_function, unicode_literals
from collections import deque
from statistics import mean
import os
import subprocess
from time import sleep
import pandas as pd
# from apscheduler.schedulers.background import BlockingScheduler
# from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
# from pytz import timezone
import argparse

from PyInquirer import style_from_dict, Token, prompt, Separator
from algorithms.MultipartiteRank import getWeightedKeyPhrasesUsingMultipartiteRank
from algorithms.PositionRank import getWeightedKeyPhrasesUsingPositionRank
from algorithms.TfIdf import getWeightedKeyPhrasesUsingTfIdf
from algorithms.YAKE import getWeightedKeyPhrasesUsingYAKE
parser = argparse.ArgumentParser()
parser.add_argument('n', nargs='?', const=0, type=int, default=1)
args = parser.parse_args()
# clear the screen
subprocess.call('cls', shell=True)

# from utils.git import handleGitCommit, handleGitPull, handleGitPush
from utils.rich_console import withLoaderDictAsParam, console, print_tree
from utils.file_utils import read_from_pickle


from text_pre_process.text_conversions import convertPdfToTextManually, convertPdfToText

from algorithms import getWeightedKeyPhrasesUsingTeKET, KeyPhrasesManager, saveKeyPhrases, saveFailedInKeyPhrasesExtraction, get_cosine_similarity, get_jaccard_similarity, getWeightedKeyPhrasesUsingTopicRank


from graph.graph_info import GraphManager, InfoManager


def runManualConversions():
    pdf_path = os.path.join(os.getcwd(), "data\\papers")
    convertPdfToTextManually(PDF_PATH=pdf_path)


def runAutomatedConversions():
    n = args.n
    pdf_path = os.path.join(os.getcwd(), "data\\papers")
    error_path = os.path.join(os.getcwd(), "data\\info")

    pdf_files_id_list = [f.name.split(".")[0] for f in os.scandir(
        pdf_path) if f.name.endswith(".pdf")]

    text_files_id_list = [f.name.split(".")[0] for f in os.scandir(
        pdf_path) if f.name.endswith(".txt")]

    text_yet_to_be_converted = list(
        set(pdf_files_id_list) - set(text_files_id_list))

    text_yet_to_be_converted_len = len(text_yet_to_be_converted)

    if n > text_yet_to_be_converted_len:
        n = text_yet_to_be_converted_len

    if n == 0:
        console.log(f"No papers to convert")
        return

    console.log(f"{text_yet_to_be_converted_len} papers needs to be converted")
    console.log(f"Converting {n} at a time")
    err_count = 0
    q = deque(text_yet_to_be_converted[:n])
    while q:
        ID = q.popleft()
        console.log(f"[yellow]Converting: {ID}[/]")
        done = withLoaderDictAsParam(convertPdfToText, {
            "ID": ID,
            "PDF_PATH": pdf_path,
            "OUTPUT_PATH": pdf_path,
            "ERROR_PATH": error_path,
            "METHOD": "pypdf2"
        }, "Converting PDF to Text")
        if done:
            console.log(f"[green]Converted:  {ID}[/]")

        else:
            console.log(f"[red]Error converting {ID}[/]")
            err_count += 1
        sleep(.5)

    console.log(f"{n-err_count} papers converted; {err_count} errors")


def runManualKeyPhraseExtractionUsingTeKET():
    clean_text_files = [os.path.join(os.getcwd(
    ), "data\\papers\\text_demo", f) for f in os.listdir("data/papers/text_demo") if f.endswith("clean.txt")]
    for path in clean_text_files:
        console.log(getWeightedKeyPhrasesUsingTeKET(path))


def runAutomatedKeyPhraseExtractionUsingTeKET():
    n = args.n
    cwd = os.getcwd()
    text_path = os.path.join(cwd, "data\\papers")
    info_path = os.path.join(cwd, "data\\info")
    full_csv_path = os.path.join(info_path, "key_phrases_extracted.csv")
    full_csv_path_failed = os.path.join(
        info_path, "key_phrases_extraction_error.csv")

    text_files_id_list = [f.name.split(".")[0] for f in os.scandir(
        "data/papers") if f.name.endswith(".txt") if os.stat(f"data/papers/{f.name}").st_size > 0]

    kp = pd.read_csv(full_csv_path, usecols=['paper_id'])
    already_extracted = kp.paper_id.unique().tolist()

    error_list = pd.read_csv(full_csv_path_failed)
    error_list = error_list.id.unique().tolist()

    processed = (set(error_list) | set(already_extracted))
    yet_to_be_extracted = list(set(text_files_id_list) - processed)
    yet_to_be_extracted_len = len(yet_to_be_extracted)

    if n > yet_to_be_extracted_len:
        n = yet_to_be_extracted_len

    if n == 0:
        console.log(f"No papers to extract key phrases")
        return

    console.log(f"{yet_to_be_extracted_len} papers needs to be extracted")
    console.log(f"Extracting {n} at a time")

    q = deque(yet_to_be_extracted[:n])
    while q:
        ID = q.popleft()
        paper_text_full_path = os.path.join(text_path, f"{ID}.txt")

        console.log(f"[yellow]Extracting: {ID}[/]")
        kps = withLoaderDictAsParam(getWeightedKeyPhrasesUsingTeKET, {
            "paper_text_full_path": paper_text_full_path,
        }, "Extracting Key Phrases", 'dots')

        if not kps:
            console.log(f"[red]No key phrases are extracted {ID}[/]")
            saveFailedInKeyPhrasesExtraction(full_csv_path_failed, ID)
            continue

        withLoaderDictAsParam(saveKeyPhrases,
                              {"df_path": full_csv_path, "key_phrases": kps, "paper_id": ID}, "Saving Key Phrases")
        console.log(f"[green]Extracted:   {ID}[/]")

    console.log(f"All {n} paper's key phrases extracted")


def runAutomatedKeyPhraseExtractionUsingOthers():
    n = args.n
    cwd = os.getcwd()
    text_path = os.path.join(cwd, "data\\papers")
    info_path = os.path.join(cwd, "data\\info")
    full_csv_path = os.path.join(info_path, "key_phrases_extracted.csv")

    """ key_phrases_extracted_topicrank.csv
        key_phrases_extracted_tfidf.csv
        key_phrases_extracted_yake.csv
        key_phrases_extracted_position_rank.csv
        key_phrases_extracted_multipartiterank.csv

     """

    topicrank_save_path = os.path.join(
        info_path, "key_phrases_extracted_topicrank.csv")

    tfidf_save_path = os.path.join(
        info_path, "key_phrases_extracted_tfidf.csv")
    yake_save_path = os.path.join(
        info_path, "key_phrases_extracted_yake.csv")
    position_rank_save_path = os.path.join(
        info_path, "key_phrases_extracted_position_rank.csv")
    multipartiterank_save_path = os.path.join(
        info_path, "key_phrases_extracted_multipartiterank.csv")

    kp = pd.read_csv(full_csv_path, usecols=['paper_id'])
    already_extracted = kp.paper_id.unique().tolist()
    # print(already_extracted)
    already_extracted_len = len(already_extracted)

    kp_done = pd.read_csv('data\info\keyphrase_done_list.csv')

    if n > already_extracted_len:
        n = already_extracted_len

    if n == 0:
        console.log(f"No papers to extract key phrases")
        return
    q = deque(already_extracted[:n])
    while q:
        ID = q.popleft()
        paper_text_full_path = os.path.join(text_path, f"{ID}.txt")
        # console.log(f"[yellow]Extracting: {ID}[/]")

        done_info = kp_done[kp_done['paper_id']
                            == ID].to_dict('records')

        if not done_info:
            continue
        done_info = done_info[0]
        topic_rank_done = done_info['TopicRank']
        tfidf_done = done_info['TfIdf']
        YAKE_done = done_info['YAKE']
        PositionRank_done = done_info['PositionRank']
        MultipartiteRank_done = done_info['MultipartiteRank']

        # print(done_info)

        if not topic_rank_done:
            kps = withLoaderDictAsParam(getWeightedKeyPhrasesUsingTopicRank, {
                "paper_text_full_path": paper_text_full_path,
            }, "Extracting Key Phrases", 'dots')
            # 1
            saveKeyPhrases(df_path=topicrank_save_path,
                           key_phrases=kps, paper_id=ID)
            # 2
            kp_done.loc[kp_done['paper_id'] == ID, "TopicRank"] = 1
            kp_done.to_csv(
                'data\info\keyphrase_done_list.csv', index=False)
            console.log(f"Extracted: {ID}")

        if not tfidf_done:
            kps = withLoaderDictAsParam(getWeightedKeyPhrasesUsingTfIdf, {
                "paper_text_full_path": paper_text_full_path,
            }, "Extracting Key Phrases", 'dots')
            saveKeyPhrases(df_path=tfidf_save_path,
                           key_phrases=kps, paper_id=ID)

            kp_done.loc[kp_done['paper_id'] == ID, "TfIdf"] = 1
            kp_done.to_csv(
                'data\info\keyphrase_done_list.csv', index=False)
            console.log(f"Extracted: {ID}")

        if not YAKE_done:
            kps = withLoaderDictAsParam(getWeightedKeyPhrasesUsingYAKE, {
                "paper_text_full_path": paper_text_full_path,
            }, "Extracting Key Phrases", 'dots')
            saveKeyPhrases(df_path=yake_save_path,
                           key_phrases=kps, paper_id=ID)

            kp_done.loc[kp_done['paper_id'] == ID, "YAKE"] = 1
            kp_done.to_csv(
                'data\info\keyphrase_done_list.csv', index=False)
            console.log(f"Extracted: {ID}")

        if not PositionRank_done:
            kps = withLoaderDictAsParam(getWeightedKeyPhrasesUsingPositionRank, {
                "paper_text_full_path": paper_text_full_path,
            }, "Extracting Key Phrases", 'dots')
            # 1
            saveKeyPhrases(df_path=position_rank_save_path,
                           key_phrases=kps, paper_id=ID)
            # 2
            kp_done.loc[kp_done['paper_id'] == ID, "PositionRank"] = 1
            kp_done.to_csv(
                'data\info\keyphrase_done_list.csv', index=False)
            console.log(f"Extracted: {ID}")

        if not MultipartiteRank_done:
            kps = withLoaderDictAsParam(getWeightedKeyPhrasesUsingMultipartiteRank, {
                "paper_text_full_path": paper_text_full_path,
            }, "Extracting Key Phrases", 'dots')
            # 1
            saveKeyPhrases(df_path=multipartiterank_save_path,
                           key_phrases=kps, paper_id=ID)
            # 2
            kp_done.loc[kp_done['paper_id'] == ID, "MultipartiteRank"] = 1
            kp_done.to_csv(
                'data\info\keyphrase_done_list.csv', index=False)
            console.log(f"Extracted: {ID}")

    console.log(f"All {n} paper's key phrases extracted")


def runFull():
    info_path = os.path.join(os.getcwd(), "data\\info")
    key_phrase_csv_path = f"{info_path}/key_phrases_extracted.csv"

    km = KeyPhrasesManager(key_phrase_csv_path)
    g_serialized = read_from_pickle('data/info/graph_serialized.dat')
    graph: GraphManager = GraphManager()
    graph = g_serialized

    console.log("GraphManager loaded")
    recommended_papers = withLoaderDictAsParam(recommendPapers, {
        "km": km,
        "im": graph.im,
        "graph": graph,
    }, "Please Wait for Recommendation", 'dots')
    console.log("~")
    print_tree(recommended_papers)
    # recommendPapers(graph=graph, im=im, km=km)


def recommendPapers(graph: GraphManager, im: InfoManager, km: KeyPhrasesManager):
    root_paper = graph.levelOrderList[0][0]
    root_paper_id = root_paper['paper_id']
    root_paper_uuid = root_paper['uuid']
    root_paper_key_phrases = km.getKeyPhrasesList(root_paper_id)
    root_paper_info = im.get_info_keys_by_uuid(root_paper_uuid, ['title'])

    selected_papers = []
    stats_log = []

    node_centrality = []

    for i, level_i in enumerate(graph.levelOrderList):
        if i == 0:
            continue
        score_list_level_i = []
        paper_count_in_level_i = len(level_i)
        score_calculated_count_in_level_i = 0

        for paper in level_i:
            paper_id = paper['paper_id']
            paper_uuid = paper['uuid']
            parent_id = paper['parent_id']
            average_list = []

            if not km.isKeyPhraseExist(paper_id):
                # console.log(f"[red]{paper_id} key phrases not exist[/]")
                continue
            paper_key_phrases = km.getKeyPhrasesList(paper_id)

            paper_info = im.get_info_keys_by_uuid(
                paper_uuid, ['title', 'freshness_avg'])
            parent_paper_id = im.get_info_keys_by_uuid(
                parent_id, ['paper_id'])['paper_id']
            # cosine similarity between parent and child
            cosine_similarity_with_root = get_cosine_similarity(
                root_paper_key_phrases, paper_key_phrases)
            average_list.append(cosine_similarity_with_root)
            # jaccard similarity between parent and child

            jaccard_similarity_with_root = get_jaccard_similarity(
                root_paper_key_phrases, paper_key_phrases)
            average_list.append(jaccard_similarity_with_root)

            if km.isKeyPhraseExist(parent_paper_id):
                parent_paper_key_phrases = km.getKeyPhrasesList(
                    parent_paper_id)
                # cosine similarity between immediate parent and child
                cosine_similarity_with_immediate_parent = get_cosine_similarity(
                    parent_paper_key_phrases, paper_key_phrases)
                average_list.append(cosine_similarity_with_immediate_parent)
                # jaccard similarity between immediate parent and child
                jaccard_similarity_with_immediate_parent = get_jaccard_similarity(
                    parent_paper_key_phrases, paper_key_phrases)
                average_list.append(jaccard_similarity_with_immediate_parent)

            freshness = paper_info['freshness_avg']
            average_list.append(freshness)

            average = mean(average_list)
            score_calculated_count_in_level_i += 1
            print(average_list)
            print(average)
            # print(f"{paper_info['title']} {average_similarity}")
            node_centrality.append([parent_id, paper_id, average])
            score_list_level_i.append(
                {**paper, 'score': average, 'title': paper_info['title']})
            # break

        score_list_level_i = sorted(
            score_list_level_i, key=lambda x: x['score'], reverse=True)

        # https://stackoverflow.com/questions/11092511/list-of-unique-dictionaries
        # drops duplicates
        unique_papers_top_n = pd.DataFrame(score_list_level_i).drop_duplicates(subset=['paper_id']).to_dict(
            'records')
        # print(unique_papers_top_n[:2])
        stats_log.append(
            [paper_count_in_level_i, score_calculated_count_in_level_i])

        selected_papers.append(unique_papers_top_n)
    # pprint(selected_papers)

    recommended_papers = [{**root_paper, "title": root_paper_info['title'],
                           'score': "~"}]
    for i, level_i in enumerate(selected_papers):
        # for paper in level_i:
        score_list_level_i = sorted(
            level_i, key=lambda x: x['score'], reverse=True)
        for paper in score_list_level_i:
            found = list(
                filter(lambda x: x['paper_id'] == paper['paper_id'], recommended_papers))
            # get papers that are not in recommended papers in upper level
            if not found:
                recommended_papers.append(paper)
                break

    # recommended_papers

    recommended_papers = [
        f"[bold green]{d['title'][:60]}...[/]|[yellow]score:{str(d['score'])[:3]}[/]" for d in recommended_papers]

    for i, s in enumerate(stats_log, start=1):  # start=1 to skip root paper
        total_papers = s[0]
        processed = s[1]
        ratio = round((processed / total_papers) * 100, 2)
        recommended_papers[i] += f"|[yellow]among:{processed}[/]|[yellow]total:{total_papers}[/]|[yellow]ratio:{ratio}%[/]"

    return recommended_papers


def App():
    custom_style_3 = style_from_dict({
        Token.Separator: '#6C6C6C',
        Token.QuestionMark: '#E91E63 bold',
        Token.Selected: '#673AB7 bold',
        Token.Instruction: '',  # default
        Token.Answer: '#2196f3 bold',
        Token.Question: '',
    })

    questions = [
        {
            'type': 'list',
            'name': 'job',
            'message': 'Use Job Scheduler?',
            'choices': [
                'No', 'Yes'
            ],
            "default": "No"
        },
        {
            'type': 'list',
            'name': 'option',
            'message': 'What do you want to do?',
            'choices': ['Convert Pdf To Text? [manual]',
                        'Convert Pdf To Text? [automated]',
                        'Run Other KeyPhrase Extract.? [automated]',
                        'Run TeKET? [manual]',
                        'Run TeKET? [automated]',
                        "Run Recommender System?"],
        }
    ]

    answers = prompt(questions, style=custom_style_3)

    if answers['job'] == 'Yes':
        # startJob(answers)
        console.print("[red]Not implemented yet[/]")
    else:
        if answers['option'] == 'Convert Pdf To Text? [manual]':
            runManualConversions()
        elif answers['option'] == 'Convert Pdf To Text? [automated]':
            runAutomatedConversions()
        elif answers['option'] == 'Run TeKET? [manual]':
            runManualKeyPhraseExtractionUsingTeKET()
        elif answers['option'] == 'Run TeKET? [automated]':
            runAutomatedKeyPhraseExtractionUsingTeKET()
        elif answers['option'] == 'Run Other KeyPhrase Extract.? [automated]':
            runAutomatedKeyPhraseExtractionUsingOthers()
        elif answers['option'] == "Run Recommender System?":
            runFull()
        else:
            print("[red]Not implemented yet[/]")


if __name__ == '__main__':
    App()
