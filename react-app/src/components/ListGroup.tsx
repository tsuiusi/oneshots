// import { Fragment } from "react";
// Can be achieved by just using <> and not importing fragment


// a component cannot return more than 1 element (listgroup can only return listgroup and not h1)
// this can be solved by wrapping the whole thing in a div or fragments
function ListGroup() {
	let items = [
		'New York',
		'Hong Kong', 
		'London',
		'Pennsylvania'
	];
	// items = [];

	const getMessage = () => {
		return items.length === 0 ? <p>empty</p> : null;
	}

	return (
		<>	
			<h1>List</h1>
			{/* {items.length === 0 && <p>empty</p>} */}
			<ul className="list-group">
				{
					// If we're retrieving items from an API it's usually a class
					// So we'll list a property (e.g item.id) instead of just item
					items.map((item, index) => (<li 
						className="list-group-item" 
						key={item} 
						onClick={() => console.log(item, index)}
					>
						{item}
					</li>))
				}
			</ul>
		</>
	);
}


export default ListGroup;
