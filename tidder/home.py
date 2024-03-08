import os

import attrs
import extra_streamlit_components as stx
import numpy as np
import pandas as pd
import streamlit as st
from bertopic import BERTopic
from streamlit_tree_select import tree_select

from tidder.models.bertopic import BERTopicDirector

DATA_PATH = "./data"
CREATOR_ID = "dankoe"
TOPIC_COLUMN_NAME = "Topic"

DEFAULT_N_RANGE = (2, 20)
MIN_N_TOPICS = 2
MAX_N_TOPICS = 100


@st.cache_resource
def build_bertopic(
    df: pd.DataFrame,
    target_column: str,
    n_range: tuple[int, int],
):
    print("Calculating topics...", target_column, n_range)
    video_bertopic = BERTopicDirector.build_bertopic(
        df,
        target_column=target_column,
        random_state=42,
        k_range=range(n_range[0], n_range[1]),
        fit=True,
        plot=False,
    )

    # video_bertopic.get_topic_info()
    return video_bertopic


@st.cache_resource
def create_tab_data(
    title: str,
    description: str,
    input_data: pd.DataFrame,
    n_range: tuple[int, int],
    _parent_topics: list["Topic"] | None = None,
    input_target_column: str = "content",
):
    return TabData(
        title=title,
        description=description,
        input_data=input_data,
        input_target_column=input_target_column,
        parent_topics=_parent_topics,
        n_range=n_range,
    )


@st.cache_data
def load_data(file_path: str):
    return pd.read_csv(file_path)


@st.cache_resource
def load_initial_tab_data(
    input_data: pd.DataFrame,
    input_target_column: str = "content",
):
    return [
        create_tab_data(
            title="Main topics",
            description="Analysis of all documents.",
            input_data=input_data,
            input_target_column=input_target_column,
            n_range=DEFAULT_N_RANGE,
        ),
    ]


@attrs.define
class Topic:
    id: str
    name: str
    textual_repr: str
    keyword_repr: str
    children: list["Topic"] | None = None
    files: list[str] | None = None

    @classmethod
    def from_topic_info(cls, row: pd.Series, documents_info: pd.DataFrame):
        return cls(
            id=str(row.name),
            name=f"Topic {str(row.name)}: `{', '.join(row['ReducedKeywords'][:5])}...`",
            textual_repr=row["Textual"],
            keyword_repr=row["ReducedKeywords"],
            files=documents_info[
                documents_info[TOPIC_COLUMN_NAME] == str(row[TOPIC_COLUMN_NAME])
            ]["video_title"]
            .unique()
            .tolist(),
        )

    def to_tree_node(self):
        node = {
            "label": self.name,
            "value": self.id,
        }
        if self.children:
            node["children"] = [child.to_tree_node() for child in self.children]

        return node


@attrs.define
class TabData:
    title: str
    description: str
    input_data: pd.DataFrame = attrs.field(repr=False)
    input_target_column: str = attrs.field(repr=False)
    n_range: tuple[int, int] = attrs.field(repr=False)
    
    parent_topics: list[Topic] | None = attrs.field(repr=False, default=None)
    
    bertopic: BERTopic = attrs.field(repr=False, init=False)
    topics: list[Topic] = attrs.field(repr=False, init=False)

    def __attrs_post_init__(self):
        # BERTopic
        self.bertopic = build_bertopic(
            self.input_data,
            self.input_target_column,
            self.n_range,
        )

        # Update input data
        topics, _ = self.bertopic.transform(self.input_data[self.input_target_column])
        self.input_data[TOPIC_COLUMN_NAME] = [str(tid) for tid in topics]

        # Create topics
        topic_info = self.bertopic.get_topic_info()
        topic_info["Textual"] = topic_info["Textual"].apply(lambda li: li[0])

        self.topics = topic_info.apply(
            lambda row: Topic.from_topic_info(row, self.input_data), axis=1
        ).tolist()


if "n_range" not in st.session_state:
    st.session_state["n_range"] = DEFAULT_N_RANGE

if "video_data" not in st.session_state:
    st.session_state["video_data"] = load_data(
        os.path.join(DATA_PATH, f"{CREATOR_ID}_videos_content.csv")
    )

if "chapters_data" not in st.session_state:
    st.session_state["chapters_data"] = load_data(
        os.path.join(DATA_PATH, f"{CREATOR_ID}_chapters_content.csv")
    )

if "tab_data" not in st.session_state:
    st.session_state["tab_data"] = load_initial_tab_data(st.session_state["video_data"])

tab_data: list[TabData] = st.session_state["tab_data"]
id_tab_map: dict[int, TabData] = {ix: tab for ix, tab in enumerate(tab_data)}

selected_tab_id = stx.tab_bar(
    data=[
        stx.TabBarItemData(id=ix, title=tab.title, description=tab.description)
        for ix, tab in id_tab_map.items()
    ],
    default=0,
    return_type=int,
)

# st.set_page_config(page_title="Content organization", page_icon="🗃")

################ SIDEBAR ################
with st.sidebar:
    st.header("About")
    st.text("Analysis of Dan Koe videos.")

    st.header("Topics")
    nodes = [topic.to_tree_node() for topic in id_tab_map[selected_tab_id].topics]
    topic_nodes = tree_select(nodes, checked=[n["value"] for n in nodes])
    st.session_state["selected_topics"] = topic_nodes["checked"]

    # Useful for debugging, adds the checked and expanded elements to the UI
    # st.write(topic_nodes)

    st.divider()

    with st.expander("Don't like the results? Change the number of topics"):
        recalculate_n_range = st.slider(
            "Number of topics",
            value=st.session_state["n_range"],
            min_value=MIN_N_TOPICS,
            max_value=MAX_N_TOPICS,
        )

        if st.button("Recalculate"):
            id_tab_map[selected_tab_id] = create_tab_data(
                title=id_tab_map[selected_tab_id].title,
                description=id_tab_map[selected_tab_id].description,
                input_data=id_tab_map[selected_tab_id].input_data,
                input_target_column=id_tab_map[selected_tab_id].input_target_column,
                _parent_topics=id_tab_map[selected_tab_id].topics,
                n_range=recalculate_n_range,
            )

    with st.expander("Want more? Explore subtopics"):
        if not topic_nodes["checked"]:
            st.warning("Select at least one topic to find subtopics.")
        elif all(
            [
                True if topic.id in topic_nodes["checked"] else False
                for topic in id_tab_map[selected_tab_id].topics
            ]
        ):
            st.info(
                "All topics are selected. No subtopics to show. Try changing the number of topics instead."
            )
        else:
            new_tab_title = st.text_input("Tab name", placeholder="Favorite videos")
            new_tab_description = st.text_area(
                "Description", placeholder="A collection of my favorite videos."
            )
            subtopics_n_range = st.slider(
                "Number of sub-topics",
                value=DEFAULT_N_RANGE,
                min_value=MIN_N_TOPICS,
                max_value=MAX_N_TOPICS,
            )

            if st.button("Calculate subtopics"):
                if not new_tab_title or not new_tab_description:
                    st.warning("Please provide a name and description for the new tab.")
                else:
                    tab_data.append(
                        create_tab_data(
                            title=new_tab_title,
                            description=new_tab_description,
                            input_data=id_tab_map[selected_tab_id]
                            .input_data[
                                id_tab_map[selected_tab_id]
                                .input_data[TOPIC_COLUMN_NAME]
                                .isin(topic_nodes["checked"])
                            ]
                            .reset_index(drop=True),
                            _parent_topics=[
                                topic
                                for topic in id_tab_map[selected_tab_id].topics
                                if topic.id in topic_nodes["checked"]
                            ],
                            n_range=subtopics_n_range,
                        )
                    )
                    st.session_state["n_range"] = subtopics_n_range
                    st.rerun()

################ METRICS ################
topic_header = "topic"
if id_tab_map[selected_tab_id].parent_topics:
    topic_header = "subtopic"

    with st.expander("Expand information about parent topics"):
        for topic in id_tab_map[selected_tab_id].parent_topics:
            st.header(topic.name)
            st.subheader("Textual explanation:")
            st.write(topic.textual_repr)

            st.subheader("Keywords:")
            st.write(topic.keyword_repr)

st.header(f"{topic_header.title()} overview")

col1, col2, col3 = st.columns(3)

topic_count = len(id_tab_map[selected_tab_id].topics)
selected_topic_count = len(st.session_state["selected_topics"]) or topic_count
col1.metric(
    f"Number of {topic_header}",
    selected_topic_count,
    (selected_topic_count - topic_count) or None,
)

file_count = [len(topic.files) for topic in id_tab_map[selected_tab_id].topics]
selected_file_count = [
    len(topic.files)
    for topic in id_tab_map[selected_tab_id].topics
    if topic.id in st.session_state["selected_topics"]
]

print(file_count)
print(selected_file_count)

col2.metric(
    "Number of files",
    sum(selected_file_count),
    (sum(selected_file_count) - sum(file_count)) or None,
)
col3.metric("Median file count per topic", np.median(selected_file_count))

################ EXTRA INFO ################
st.header(f"{topic_header.title()}s' details")

for topic in id_tab_map[selected_tab_id].topics:
    if (
        st.session_state["selected_topics"]
        and topic.id not in st.session_state["selected_topics"]
    ):
        continue

    with st.expander(f"Expand {topic.name}"):
        st.subheader("Textual explanation:")
        st.write(topic.textual_repr)

        st.subheader("Keywords:")
        st.write(topic.keyword_repr)

        st.subheader("Files:")
        if topic.files:
            st.write(topic.files)
        else:
            st.write("Wow, such empty.")


if hasattr(id_tab_map[selected_tab_id].bertopic.hdbscan_model, "plot_fig"):
    with st.expander(f"Show {topic_header}s' inertia scores"):
        st.pyplot(id_tab_map[selected_tab_id].bertopic.hdbscan_model.plot_fig)

# Reference for more styling https://discuss.streamlit.io/t/add-tab-dynamically/28541
