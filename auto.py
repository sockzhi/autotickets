from playwright.sync_api import sync_playwright, Playwright
from playwright.sync_api import Frame, Page
from groq import Groq
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup
import os
import copy
import structlog
import hashlib
import json
import time
import re

ACTIONS = {
    'CLICK': 'CLICK',
    'INPUT_TEXT': 'INPUT_TEXT',
    'SELECT_OPTION': 'SELECT_OPTION',
    'CHECKBOX': 'CHECKBOX',
    'WAIT': 'WAIT',
    'NULL_ACTION': 'NULL_ACTION',
    'TERMINATE': 'TERMINATE',
    'COMPLETE': 'COMPLETE'
}

api_key=""
ID_ATTR: str = "unique_id"
LOG = structlog.get_logger()
def generate_chat_completion(user_message):
    # client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_message}],
        model="llama3-70b-8192",
    )
    # Regular expression pattern to match the JSON block
    # json_pattern = r'\{.*?\}'

    # # Use re.search() to extract the JSON string
    # match = re.search(json_pattern, chat_completion.choices[0].message.content, re.DOTALL)
    # json_str = ''
    # if match:
    #     json_str = match.group(0)  # Extracted JSON string
    # return json_str
    return chat_completion.choices[0].message.content

def load_js_script() -> str:
    # TODO: Handle file location better. This is a hacky way to find the file location.
    path = f"domUtils.js"
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError as e:
        LOG.exception("Failed to load the JS script", path=path)
        raise e

env = Environment(loader=FileSystemLoader(os.path.abspath(".")))
Any = object()
def load_prompt(template: str, **kwargs: Any) -> str:
    """
    Load and populate the specified template.

    Args:
        template (str): The name of the template to load.
        **kwargs: The arguments to populate the template with.

    Returns:
        str: The populated template.
    """
    template = "/".join(["prompts", template])
    jinja_template = env.get_template(f"{template}.j2")
    return jinja_template.render(**kwargs)

def build_element_dict(
    elements: list[dict],
) -> tuple[dict[str, str], dict[str, dict], dict[str, str]]:
    id_to_css_dict: dict[str, str] = {}
    id_to_element_dict: dict[str, dict] = {}
    id_to_frame_dict: dict[str, str] = {}

    for element in elements:
        element_id: str = element.get("id", "")
        # get_interactable_element_tree marks each interactable element with a unique_id attribute
        id_to_css_dict[element_id] = f"[{ID_ATTR}='{element_id}']"
        id_to_element_dict[element_id] = element
        id_to_frame_dict[element_id] = element["frame"]

    return id_to_css_dict, id_to_element_dict, id_to_frame_dict

def get_element_by_id(element_id: str, id_to_css_dict, id_to_element_dict, id_to_frame_dict) -> tuple[str, dict, str]:
    element = id_to_element_dict.get(element_id)

    frame = id_to_frame_dict.get(element_id)

    css = id_to_css_dict.get(element_id)

    return css, element, frame
        
ELEMENT_NODE_ATTRIBUTES = {
    "id",
}
# function to convert JSON element to HTML
def build_attribute(key: str, value: Any) -> str:
    if isinstance(value, bool) or isinstance(value, int):
        return f'{key}="{str(value).lower()}"'

    return f'{key}="{str(value)}"' if value else key

def json_to_html(element: dict, need_addition_attrs: bool = True) -> str:
    """
    if element is flagged as dropped, the html format is empty
    """
    if element.get("isDropped", False):
        return ""

    attributes: dict[str, Any] = copy.deepcopy(element.get("attributes", {}))

    if need_addition_attrs:
        # adding the node attribute to attributes
        for attr in ELEMENT_NODE_ATTRIBUTES:
            value = element.get(attr)
            if value is None:
                continue
            attributes[attr] = value

    attributes_html = " ".join(build_attribute(key, value) for key, value in attributes.items())

    tag = element["tagName"]
    if element.get("isSelectable", False):
        tag = "select"

    text = element.get("text", "")
    # build children HTML
    children_html = "".join(json_to_html(child) for child in element.get("children", []))
    # build option HTML
    option_html = "".join(
        f'<option index="{option.get("optionIndex")}">{option.get("text")}</option>'
        for option in element.get("options", [])
    )

    if element.get("purgeable", False):
        return children_html + option_html

    before_pseudo_text = element.get("beforePseudoText") or ""
    after_pseudo_text = element.get("afterPseudoText") or ""

    # Check if the element is self-closing
    if (
        tag in ["img", "input", "br", "hr", "meta", "link"]
        and not option_html
        and not children_html
        and not before_pseudo_text
        and not after_pseudo_text
    ):
        return f'<{tag}{attributes_html if not attributes_html else " "+attributes_html}>'
    else:
        return f'<{tag}{attributes_html if not attributes_html else " "+attributes_html}>{before_pseudo_text}{text}{children_html+option_html}{after_pseudo_text}</{tag}>'

def _remove_rect(element: dict) -> None:
    if "rect" in element:
        del element["rect"]

def _get_svg_cache_key(hash: str) -> str:
    return f"zh:svg:{hash}"

def _get_shape_cache_key(hash: str) -> str:
    return f"zh:shape:{hash}"

USELESS_SHAPE_ATTRIBUTE = [ID_ATTR, "id", "aria-describedby"]
SVG_MAX_LENGTH: int = 100000
INVALID_SHAPE = "N/A"
def _remove_addition_attributes(element: dict) -> dict:
    """
    To get the original HTML element without additional attributes
    """
    element_copied = copy.deepcopy(element)
    for attr in ELEMENT_NODE_ATTRIBUTES:
        if element_copied.get(attr):
            del element_copied[attr]

    if "attributes" in element_copied:
        attributes: dict = copy.deepcopy(element_copied.get("attributes", {}))
        for key in attributes.keys():
            if key in USELESS_SHAPE_ATTRIBUTE:
                del element_copied["attributes"][key]

    children: list[dict] | None = element_copied.get("children", None)
    if children is None:
        return element_copied

    trimmed_children = []
    for child in children:
        trimmed_children.append(_remove_addition_attributes(child))

    element_copied["children"] = trimmed_children
    return element_copied

def _convert_svg_to_string(
    element: dict,
) -> None:
    if element.get("tagName") != "svg":
        return

    if element.get("isDropped", False):
        return

    element_id = element.get("id", "")
    svg_element = _remove_addition_attributes(element)
    svg_html = json_to_html(svg_element)
    hash_object = hashlib.sha256()
    hash_object.update(svg_html.encode("utf-8"))
    svg_hash = hash_object.hexdigest()
    svg_key = _get_svg_cache_key(svg_hash)

    svg_shape: str | None = None

    if svg_shape:
        LOG.debug("SVG loaded from cache", element_id=element_id, key=svg_key, shape=svg_shape)
    else:
        if len(svg_html) > SVG_MAX_LENGTH:
            # TODO: implement a fallback solution for "too large" case, maybe convert by screenshot
            LOG.warning(
                "SVG element is too large to convert, going to drop the svg element.",
                element_id=element_id,
                # task_id=task_id,
                # step_id=step_id,
                length=len(svg_html),
                key=svg_key,
            )
            del element["children"]
            element["isDropped"] = True
            return

        LOG.debug("call LLM to convert SVG to string shape", element_id=element_id)
        svg_convert_prompt = load_prompt("svg-convert", svg_element=svg_html)

        for retry in range(1):
            # try:
            # print(svg_convert_prompt)
            # json_response = generate_chat_completion(user_message=svg_convert_prompt)
            # print(json_response)
            # # time.sleep(5)
            # json_response = json.loads(json_response)
            # # print(json_response)
            # svg_shape = json_response.get("shape", "")
            # # print(svg_shape)
            # recognized = json_response.get("recognized", False)
            # if not svg_shape or not recognized:
            #     raise Exception("Empty or unrecognized SVG shape replied by secondary llm")
            LOG.info("SVG converted by LLM", element_id=element_id, key=svg_key, shape=svg_shape)
            break
        else:
            LOG.warning(
                "Reaching the max try to convert svg element, going to drop the svg element.",
                element_id=element_id,
                # task_id=task_id,
                # step_id=step_id,
                key=svg_key,
                length=len(svg_html),
            )
            del element["children"]
            element["isDropped"] = True
            return

    element["attributes"] = dict()
    del element["children"]
    return

def _convert_css_shape_to_string(
    frame: Page | Frame,
    element: dict,
) -> None:
    element_id: str = element.get("id", "")

    # task_id = task.task_id if task else None
    # step_id = step.step_id if step else None
    shape_element = _remove_addition_attributes(element)
    svg_html = json_to_html(shape_element)
    hash_object = hashlib.sha256()
    hash_object.update(svg_html.encode("utf-8"))
    shape_hash = hash_object.hexdigest()
    shape_key = _get_shape_cache_key(shape_hash)

    css_shape: str | None = None

    if css_shape:
        LOG.debug("CSS shape loaded from cache", element_id=element_id, key=shape_key, shape=css_shape)
    else:
        # FIXME: support element in iframe
        locater = frame.locator(f'[{ID_ATTR}="{element_id}"]')
        if locater.count() == 0:
            LOG.info(
                "No locater found to convert css shape",
                element_id=element_id,
            )
            return None

        if locater.count() > 1:
            LOG.info(
                "multiple locaters found to convert css shape",
                element_id=element_id,
            )
            return None

        try:
            LOG.debug("call LLM to convert css shape to string shape", element_id=element_id)
            screenshot = locater.screenshot(timeout=20000)
            prompt = load_prompt("css-shape-convert")

            # TODO: we don't retry the css shape conversion today
            for retry in range(0):
                # try:
                json_response = generate_chat_completion(user_message=prompt) #TODO need add screenshot later
                css_shape = json_response.get("shape", "")
                recognized = json_response.get("recognized", False)
                if not css_shape or not recognized:
                    raise Exception("Empty or unrecognized css shape replied by secondary llm")
                LOG.info("CSS Shape converted by LLM", element_id=element_id, key=shape_key, shape=css_shape)
                break
            else:
                LOG.info(
                    "Max css shape convertion retry, going to abort the convertion.",
                    # task_id=task_id,
                    # step_id=step_id,
                    element_id=element_id,
                    key=shape_key,
                )
                return None
        except Exception:
            LOG.warning(
                "Failed to convert css shape to string shape by LLM",
                key=shape_key,
                # task_id=task_id,
                # step_id=step_id,
                element_id=element_id,
                exc_info=True,
            )
            return None

    if "attributes" not in element:
        element["attributes"] = dict()
    # if css_shape != INVALID_SHAPE:
    #     # refresh the cache expiration
    #     await app.CACHE.set(shape_key, css_shape)
    #     element["attributes"]["shape-description"] = css_shape
    return None

def _should_css_shape_convert(element: dict) -> bool:
    if "id" not in element:
        return False

    tag_name = element.get("tagName")
    if tag_name not in ["a", "span", "i"]:
        return False

    # should be without children
    if len(element.get("children", [])) > 0:
        return False

    # should be no text
    if element.get("text"):
        return False

    # if <span> and <i>  we try to convert the shape
    if tag_name in ["span", "i"]:
        return True

    # if <a>, it should be no text, no href/target attribute
    if tag_name == "a":
        attributes = element.get("attributes", {})
        if "href" in attributes:
            return False

        if "target" in attributes:
            return False
        return True

    return False

def cleanup_element_tree_func(frame: Page | Frame, url: str, element_tree: list[dict]) -> list[dict]:
    """
    Remove rect and attribute.unique_id from the elements.
    The reason we're doing it is to
    1. reduce unnecessary data so that llm get less distrction
    TODO later: 2. reduce tokens sent to llm to save money
    :param elements: List of elements to remove xpaths from.
    :return: List of elements without xpaths.
    """
    queue = []
    for element in element_tree:
        queue.append(element)
    while queue:
        queue_ele = queue.pop(0)
        _remove_rect(queue_ele)
        _convert_svg_to_string(queue_ele)

        # if _should_css_shape_convert(element=queue_ele):
        #     _convert_css_shape_to_string(
        #         frame=frame,
        #         element=queue_ele,
        #         # task=task,
        #         # step=step,
        #     )

        # TODO: we can come back to test removing the unique_id
        # from element attributes to make sure this won't increase hallucination
        # _remove_unique_id(queue_ele)
        if "children" in queue_ele:
            queue.extend(queue_ele["children"])
    return element_tree

url = "https://www.booking.com/flights"
navigation_goal = "You already on flight search page, search a One-way ticket.Enter 'ROC' as the departure airport, 'LAX' as the destination airport, and '12/28/2024' as the travel date. Click the search button to display available flights. COMPLETE when the list of available flights is displayed on the screen."
def run(playwright: Playwright):
    print("GOAL:", navigation_goal)
    webkit = playwright.webkit
    browser = webkit.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    draw_boxes = True
    JS_FUNCTION_DEFS = load_js_script()
    # print(JS_FUNCTION_DEFS)
    page.evaluate(expression=JS_FUNCTION_DEFS)
    #bounding box
    js_script = f"() => scrollToTop({str(draw_boxes).lower()})"
    page.evaluate(expression=js_script)
    #screen shot
    page.screenshot(path="screenshot.png")
    #get element tree
    main_frame_js_script = "() => buildTreeFromBody()"
    elements, element_tree = page.evaluate(expression=main_frame_js_script)
    element_tree = cleanup_element_tree_func(page, url, copy.deepcopy(element_tree))
    id_to_css_dict, id_to_element_dict, id_to_frame_dict = build_element_dict(
        elements
    )

    # print(id_to_frame_dict)
    #get main frame
    js_script = "() => document.body.innerText"
    text = page.main_frame.evaluate(expression=js_script)
    #get html
    html = page.content()
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    # Find all SVG tags
    svgs = soup.find_all('svg')

    # Loop through each svg tag
    for svg in svgs:
        # Keep only the attributes of the <svg> tag, removing its content
        svg.clear()  # This clears all the child elements within the <svg>
    
    # Remove all <script> tags
    for script in soup.find_all('script'):
        script.decompose()
    # List of tags to keep (interactable elements)
    interactable_tags = ['svg']

    for script in soup.find_all('style'):
        script.decompose()
    # List of tags to keep (interactable elements)
    interactable_tags = ['span']

    for tag in soup.find_all(True):
        if not tag.get_text(strip=True):  # If there is no content inside the tag
            tag.decompose()  # Remove the tag
            # Remove the class attribute
    for tag in soup.find_all(True):
        if tag and tag.has_attr('class'):
            del tag['class']
    # Find all interactable elements from the list of tags
    interactable_elements = soup.find_all(interactable_tags)

    element_tree_in_prompt = "".join(json_to_html(element) for element in element_tree)
    soup2 = BeautifulSoup(element_tree_in_prompt, 'html.parser')
    for tag in soup2.find_all(True):
        if not tag.get_text(strip=True):  # If there is no content inside the tag
            tag.decompose()  # Remove the tag
    # print(soup)
    prompt = load_prompt(template="extract-action", navigation_goal=navigation_goal, current_url=url, elements=str(interactable_elements))
    action_output = generate_chat_completion(prompt)

    print(action_output)
    json_response = json.loads(action_output)
    for i in range(len(json_response["actions"])):
        css, element, frame = get_element_by_id(json_response["actions"][i]["id"], id_to_css_dict, id_to_element_dict, id_to_frame_dict)
        # print(css)
        locator = page.locator(css)
        if json_response["actions"][i]["action_type"] == ACTIONS["CLICK"]:
            locator.click(timeout=1000)
        elif json_response["actions"][i]["action_type"] == ACTIONS["INPUT_TEXT"]:
            locator.fill(json_response["actions"][i]["text"])
        # getattr(locator, ACTIONS[json_response["actions"][i]["action_type"]])()
        # locator.click(timeout=1000)
        time.sleep(5)
    browser.close()

with sync_playwright() as playwright:
    run(playwright)